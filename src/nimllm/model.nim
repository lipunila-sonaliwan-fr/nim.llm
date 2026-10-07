# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026 (adaptation)
# Portions dérivées de ggml / llama.cpp
# The ggml authors (© 2023-2026) - LICENSES/MIT-ggml.txt
# nimllm/model — moteur d'inférence Transformer (architectures llama,
# mistral, qwen2, qwen3) écrit entièrement en Nim.
#
# * `LlmModel` : poids (mappés depuis le GGUF) + tokeniseur + configuration ;
# * `LlmContext` : état d'une conversation (cache clé/valeur, tampons).
#
# Plusieurs contextes peuvent partager le même modèle.

import std/[math, tables, strutils, os, times]
import gguf, quant, tokenizer, parallel

type
  ModelConfig* = object
    # Hyper-paramètres de l'architecture.
    arch*: string                # "llama", "qwen2", "qwen3"...
    name*: string
    dim*: int                    # taille des vecteurs cachés (embedding_length).
    hidden*: int                 # taille intermédiaire du FFN.
    nLayers*: int
    nHeads*: int
    nKvHeads*: int               # < nHeads => attention groupée (GQA).
    headDim*: int
    vocab*: int
    ctxTrain*: int               # contexte maximal d'entraînement.
    ropeBase*: float
    ropeDim*: int
    ropeNeox*: bool              # style NeoX (moitiés) au lieu de paires adjacentes.
    normEps*: float
    tiedEmbeddings*: bool        # la tête de sortie réutilise token_embd.

  LoraPair* = ref object
    # Adaptateur LoRA appliqué à la volée.
    rank*: int
    a*: seq[float32]             # [rank, cols]
    b*: seq[float32]             # [rows, rank]
    scale*: float32

  QMatrix* = object
    # Matrice de poids (lignes quantifiées), généralement mappée depuis le GGUF.
    name*: string
    typ*: GgmlType
    rows*, cols*: int
    rb*: int
    data*: ptr UncheckedArray[uint8]
    owned: seq[uint8]
    lora*: LoraPair

  Layer* = object
    attnNorm*, ffnNorm*: seq[float32]
    wq*, wk*, wv*, wo*: QMatrix
    wGate*, wUp*, wDown*: QMatrix
    bq*, bk*, bv*: seq[float32]
    qNorm*, kNorm*: seq[float32]

  LlmModel* = ref object
    # Modèle chargé (poids en lecture seule, partageables entre contextes).
    path*: string
    cfg*: ModelConfig
    gguf*: GgufFile
    tokenizer*: Tokenizer
    tokEmb*, output*: QMatrix
    outNorm*: seq[float32]
    layers*: seq[Layer]
    invFreq*: seq[float32]

  LlmContext* = ref object
    # État d'évaluation : cache KV et tampons de travail.
    model*: LlmModel
    nCtx*: int
    nBatch*: int
    tokens*: seq[int]            # tokens actuellement présents dans le cache.
    kCache, vCache: seq[float32]
    x, xb, xb2, q, k, v, att, hb, hb2: seq[float32]
    logits*: seq[float32]        # logits du dernier token évalué.
    lastHidden*: seq[float32]    # états cachés finaux (normalisés) du dernier lot.
    evalTime*: float             # secondes cumulées passées dans eval.
    evalTokens*: int

  ModelError* = object of CatchableError

#
# Matrices quantifiées
#

proc rowPtr*(m: QMatrix; r: int): pointer {.inline.} = addr m.data[r * m.rb]
proc rowPtr*(m: ptr QMatrix; r: int): pointer {.inline.} = addr m.data[r * m.rb]

proc newQMatrix*(name: string; typ: GgmlType; rows, cols: int; values: openArray[float32]): QMatrix =
  # Crée une matrice en mémoire à partir de valeurs float32 [rows, cols].
  result = QMatrix(name: name, typ: typ, rows: rows, cols: cols, rb: rowBytes(typ, cols))
  result.owned = newSeq[uint8](result.rb * rows)
  for r in 0 ..< rows:
    quantizeRow(typ, unsafeAddr values[r*cols], addr result.owned[r*result.rb], cols)
  result.data = cast[ptr UncheckedArray[uint8]](addr result.owned[0])

proc qmat(g: GgufFile; name: string; optional = false): QMatrix =
  if name notin g.tensors:
    if optional: return QMatrix(name: name)
    raise newException(ModelError, "tenseur manquant : " & name)
  let ti = g.tensors[name]
  if ti.typ notin supportedTypes:
    raise newException(ModelError, "format " & $ti.typ & " non pris en charge (" & name & ")")
  result = QMatrix(name: name, typ: ti.typ, cols: ti.dims[0],
                   rows: (if ti.dims.len > 1: ti.numElements div ti.dims[0] else: 1))
  result.rb = rowBytes(ti.typ, result.cols)
  result.data = cast[ptr UncheckedArray[uint8]](ti.data)

proc isEmpty*(m: QMatrix): bool = m.data == nil

proc vec(g: GgufFile; name: string; optional = false): seq[float32] =
  if name notin g.tensors:
    if optional: return @[]
    raise newException(ModelError, "tenseur manquant : " & name)
  g.tensorF32(name)

proc dequantRowTo*(m: QMatrix; r: int; dst: var openArray[float32]) =
  dequantRow(m.typ, m.rowPtr(r), addr dst[0], m.cols)

# tampons par thread

var gScratch: seq[seq[float32]]

proc ensureScratch(n: int) =
  let nt = numThreads()
  if gScratch.len < nt: gScratch.setLen(nt)
  for s in gScratch.mitems:
    if s.len < n: s.setLen(n)

proc scratch(worker, n: int): ptr float32 {.inline.} =
  {.cast(gcsafe).}:
    addr gScratch[worker][0]

type MatCtx = object
  w: ptr QMatrix
  x: ptr UncheckedArray[float32]
  y: ptr UncheckedArray[float32]
  act: ptr QAct
  nb: int

{.push checks: off, boundChecks: off, overflowChecks: off.}
proc matmulTask(ctx: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  {.cast(gcsafe).}:
    let c = cast[ptr MatCtx](ctx)
    let w = c.w
    let cols = w.cols
    let rows = w.rows
    if c.act != nil:
      # chemin entier : activation quantifiée en int8.
      for r in first ..< last:
        let rp = w.rowPtr(r)
        for b in 0 ..< c.nb:
          c.y[b*rows + r] = dotRowAct(w.typ, rp, c.act, b*cols, cols)
      return
    if c.nb == 1 and w.typ == gtF32:
      for r in first ..< last:
        c.y[r] = dotF32(cast[ptr float32](w.rowPtr(r)), addr c.x[0], cols)
      return
    let tmp = scratch(worker, cols)
    for r in first ..< last:
      var rowp: ptr float32
      if w.typ == gtF32:
        rowp = cast[ptr float32](w.rowPtr(r))
      else:
        dequantRow(w.typ, w.rowPtr(r), tmp, cols)
        rowp = tmp
      for b in 0 ..< c.nb:
        c.y[b*rows + r] = dotF32(rowp, addr c.x[b*cols], cols)
{.pop.}

var exactMatmul* = false
  # true : désactive la quantification 8 bits des activations (calcul en
  # float32 exact, plus lent) pour la vérification ou l'entraînement.

var gAct: QAct

proc matmul*(w: QMatrix; x: ptr float32; y: ptr float32; nb: int) =
  # y[nb, rows] = x[nb, cols] · w^t  (+ LoRA éventuel).
  ensureScratch(w.cols)
  var c = MatCtx(w: unsafeAddr w, x: cast[ptr UncheckedArray[float32]](x),
                 y: cast[ptr UncheckedArray[float32]](y), nb: nb)
  let bl = actBlockLen(w.typ)
  if not exactMatmul and bl > 0 and w.cols mod bl == 0 and (nb <= 4 or w.typ != gtQ8_0):
    quantizeAct(x, nb * w.cols, bl, gAct)
    c.act = addr gAct
  parallelFor(w.rows, matmulTask, addr c, minChunk = 8)
  if w.lora != nil:
    let L = w.lora
    let xs = cast[ptr UncheckedArray[float32]](x)
    let ys = cast[ptr UncheckedArray[float32]](y)
    var tmp = newSeq[float32](L.rank)
    for b in 0 ..< nb:
      for i in 0 ..< L.rank:
        tmp[i] = dotF32(unsafeAddr L.a[i*w.cols], addr xs[b*w.cols], w.cols)
      for r in 0 ..< w.rows:
        var s = 0'f32
        for i in 0 ..< L.rank: s += L.b[r*L.rank + i] * tmp[i]
        ys[b*w.rows + r] += L.scale * s

#
# Chargement
#

proc archKey(g: GgufFile; arch, key: string; default = 0): int =
  g.getInt(arch & "." & key, default)

proc loadModel*(path: string; threads = 0; verbose = false): LlmModel =
  # Charge un modèle GGUF (llama 3.x, mistral, qwen2/2.5, qwen3...).
  # `threads` : nombre de threads de calcul (0 = tous les cœurs).
  let t0 = epochTime()
  if threads != 0 or numThreads() == 1: setThreads(threads)
  let g = openGguf(path)
  result = LlmModel(path: path, gguf: g)
  var arch = g.getStr("general.architecture", "llama")
  var cfg = ModelConfig(arch: arch, name: g.getStr("general.name", path.extractFilename))
  if arch notin ["llama", "mistral", "qwen2", "qwen3", "granite"]:
    raise newException(ModelError, "architecture non prise en charge : " & arch &
      " (pris en charge : llama, mistral, qwen2, qwen3)")
  cfg.dim = g.archKey(arch, "embedding_length")
  cfg.hidden = g.archKey(arch, "feed_forward_length")
  cfg.nLayers = g.archKey(arch, "block_count")
  cfg.nHeads = g.archKey(arch, "attention.head_count")
  cfg.nKvHeads = g.archKey(arch, "attention.head_count_kv", cfg.nHeads)
  cfg.headDim = g.archKey(arch, "attention.key_length", cfg.dim div max(1, cfg.nHeads))
  cfg.ctxTrain = g.archKey(arch, "context_length", 2048)
  cfg.ropeBase = g.getFloat(arch & ".rope.freq_base", 10000.0)
  cfg.ropeDim = g.archKey(arch, "rope.dimension_count", cfg.headDim)
  cfg.normEps = g.getFloat(arch & ".attention.layer_norm_rms_epsilon", 1e-5)
  cfg.ropeNeox = arch in ["qwen2", "qwen3"]
  result.tokenizer = tokenizerFromGguf(g)
  cfg.vocab = g.archKey(arch, "vocab_size", result.tokenizer.vocabSize)
  result.tokEmb = qmat(g, "token_embd.weight")
  cfg.vocab = result.tokEmb.rows
  result.output = qmat(g, "output.weight", optional = true)
  if result.output.isEmpty:
    result.output = result.tokEmb
    cfg.tiedEmbeddings = true
  result.outNorm = vec(g, "output_norm.weight")
  for l in 0 ..< cfg.nLayers:
    let p = "blk." & $l & "."
    var L: Layer
    L.attnNorm = vec(g, p & "attn_norm.weight")
    L.ffnNorm = vec(g, p & "ffn_norm.weight")
    L.wq = qmat(g, p & "attn_q.weight")
    L.wk = qmat(g, p & "attn_k.weight")
    L.wv = qmat(g, p & "attn_v.weight")
    L.wo = qmat(g, p & "attn_output.weight")
    L.wGate = qmat(g, p & "ffn_gate.weight")
    L.wUp = qmat(g, p & "ffn_up.weight")
    L.wDown = qmat(g, p & "ffn_down.weight")
    L.bq = vec(g, p & "attn_q.bias", true)
    L.bk = vec(g, p & "attn_k.bias", true)
    L.bv = vec(g, p & "attn_v.bias", true)
    L.qNorm = vec(g, p & "attn_q_norm.weight", true)
    L.kNorm = vec(g, p & "attn_k_norm.weight", true)
    result.layers.add L
  if cfg.hidden == 0 and cfg.nLayers > 0: cfg.hidden = result.layers[0].wUp.rows
  # fréquences RoPE (avec facteurs d'échelle Llama 3 éventuels)
  let factors = vec(g, "rope_freqs.weight", true)
  let half = cfg.ropeDim div 2
  result.invFreq = newSeq[float32](half)
  for i in 0 ..< half:
    var f = pow(cfg.ropeBase, -2.0 * i.float / cfg.ropeDim.float)
    if factors.len == half: f /= factors[i].float
    result.invFreq[i] = f.float32
  result.cfg = cfg
  if verbose:
    stderr.writeLine "[nimllm] modèle ", cfg.name, " (", arch, ") : ", cfg.nLayers,
      " couches, dim ", cfg.dim, ", vocab ", cfg.vocab, ", ", numThreads(),
      " threads, chargé en ", formatFloat(epochTime() - t0, ffDecimal, 2), " s"

proc close*(m: LlmModel) =
  # Libère le fichier mappé (le modèle devient inutilisable).
  m.gguf.close()

proc paramCount*(m: LlmModel): int =
  # Nombre total de paramètres.
  for name, ti in m.gguf.tensors: result += ti.numElements

proc describe*(m: LlmModel): string =
  let c = m.cfg
  "Modèle " & c.name & " [" & c.arch & "]\n" &
  "  couches=" & $c.nLayers & " dim=" & $c.dim & " ffn=" & $c.hidden &
  " têtes=" & $c.nHeads & " têtesKV=" & $c.nKvHeads & " dimTête=" & $c.headDim & "\n" &
  "  vocab=" & $c.vocab & " contexte=" & $c.ctxTrain & " rope_base=" & $c.ropeBase &
  " paramètres=" & formatFloat(m.paramCount.float / 1e6, ffDecimal, 1) & " M\n" &
  "  poids : " & $m.layers[0].wq.typ & " (attention), " & $m.output.typ & " (sortie)"

#
# Contexte
#

proc newContext*(m: LlmModel; nCtx = 2048; nBatch = 32): LlmContext =
  # Crée un contexte d'évaluation. `nCtx` : nombre maximal de tokens mémorisés.
  let c = m.cfg
  let n = if nCtx <= 0: c.ctxTrain else: min(nCtx, c.ctxTrain)
  result = LlmContext(model: m, nCtx: n, nBatch: max(1, nBatch))
  let kvDim = c.nKvHeads * c.headDim
  result.kCache = newSeq[float32](c.nLayers * n * kvDim)
  result.vCache = newSeq[float32](c.nLayers * n * kvDim)
  let B = result.nBatch
  let qDim = c.nHeads * c.headDim
  result.x = newSeq[float32](B * c.dim)
  result.xb = newSeq[float32](B * max(c.dim, qDim))
  result.xb2 = newSeq[float32](B * c.dim)
  result.q = newSeq[float32](B * qDim)
  result.k = newSeq[float32](B * kvDim)
  result.v = newSeq[float32](B * kvDim)
  result.att = newSeq[float32](B * qDim)
  result.hb = newSeq[float32](B * c.hidden)
  result.hb2 = newSeq[float32](B * c.hidden)
  result.logits = newSeq[float32](c.vocab)

proc reset*(ctx: LlmContext) =
  # Vide le cache (nouvelle conversation).
  ctx.tokens.setLen(0)

proc truncate*(ctx: LlmContext; n: int) =
  # Ne garde que les `n` premiers tokens du cache.
  if n < ctx.tokens.len: ctx.tokens.setLen(max(0, n))

proc nPast*(ctx: LlmContext): int = ctx.tokens.len

proc memoryUsage*(ctx: LlmContext): int =
  # Octets occupés par le cache KV.
  (ctx.kCache.len + ctx.vCache.len) * 4

# calcul

{.push checks: off, boundChecks: off, overflowChecks: off.}

proc rmsnorm(o: ptr UncheckedArray[float32]; x: ptr UncheckedArray[float32];
             w: openArray[float32]; n: int; eps: float) =
  var ss = 0.0
  for i in 0 ..< n: ss += float(x[i]) * float(x[i])
  let s = float32(1.0 / sqrt(ss / n.float + eps))
  for i in 0 ..< n: o[i] = x[i] * s * w[i]

proc rope(v: ptr UncheckedArray[float32]; nHeads, headDim, ropeDim: int; pos: int;
          invFreq: seq[float32]; neox: bool) =
  let half = ropeDim div 2
  for h in 0 ..< nHeads:
    let base = h * headDim
    for i in 0 ..< half:
      let theta = pos.float32 * invFreq[i]
      let c = cos(theta)
      let s = sin(theta)
      let (a, b) = if neox: (base + i, base + i + half) else: (base + 2*i, base + 2*i + 1)
      let x0 = v[a]
      let x1 = v[b]
      v[a] = x0 * c - x1 * s
      v[b] = x0 * s + x1 * c

type AttnCtx = object
  # uniquement des valeurs simples et des pointeurs bruts : aucun compteur de
  # références n'est manipulé depuis les threads de travail.
  q, att, kc, vc: ptr UncheckedArray[float32]
  nHeads, nKvHeads, hd, nCtx, layer, startPos: int

proc attnTask(p: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  {.cast(gcsafe).}:
    let a = cast[ptr AttnCtx](p)
    let hd = a.hd
    let kvDim = a.nKvHeads * hd
    let qDim = a.nHeads * hd
    let groups = a.nHeads div a.nKvHeads
    let scale = 1'f32 / sqrt(hd.float32)
    let loff = a.layer * a.nCtx * kvDim
    let sc = cast[ptr UncheckedArray[float32]](scratch(worker, a.nCtx))
    for job in first ..< last:
      let t = job div a.nHeads
      let h = job mod a.nHeads
      let pos = a.startPos + t
      let kvh = h div groups
      let qp = addr a.q[t*qDim + h*hd]
      var mx = -Inf.float32
      for j in 0 .. pos:
        let s = dotF32(qp, addr a.kc[loff + j*kvDim + kvh*hd], hd) * scale
        sc[j] = s
        if s > mx: mx = s
      var sum = 0'f32
      for j in 0 .. pos:
        sc[j] = exp(sc[j] - mx)
        sum += sc[j]
      let inv = 1'f32 / sum
      let o = cast[ptr UncheckedArray[float32]](addr a.att[t*qDim + h*hd])
      for i in 0 ..< hd: o[i] = 0
      for j in 0 .. pos:
        let w = sc[j] * inv
        let vp = cast[ptr UncheckedArray[float32]](addr a.vc[loff + j*kvDim + kvh*hd])
        for i in 0 ..< hd: o[i] += w * vp[i]

{.pop.}

proc ua(s: var seq[float32]; off = 0): ptr UncheckedArray[float32] {.inline.} =
  cast[ptr UncheckedArray[float32]](addr s[off])

proc evalBatch(ctx: LlmContext; toks: openArray[int]; wantAll: bool;
               allLogits: var seq[float32]) =
  let m = ctx.model
  let c = m.cfg
  let B = toks.len
  let startPos = ctx.tokens.len
  let hd = c.headDim
  let qDim = c.nHeads * hd
  let kvDim = c.nKvHeads * hd
  ensureScratch(max(max(c.dim, c.hidden), max(ctx.nCtx, qDim)))
  # embeddings.
  for b in 0 ..< B:
    let id = toks[b]
    if id < 0 or id >= c.vocab:
      raise newException(ModelError, "identifiant de token invalide : " & $id)
    dequantRow(m.tokEmb.typ, m.tokEmb.rowPtr(id), addr ctx.x[b*c.dim], c.dim)
  for l in 0 ..< c.nLayers:
    let L = m.layers[l]
    # attention.
    for b in 0 ..< B:
      rmsnorm(ua(ctx.xb, b*c.dim), ua(ctx.x, b*c.dim), L.attnNorm, c.dim, c.normEps)
    matmul(L.wq, addr ctx.xb[0], addr ctx.q[0], B)
    matmul(L.wk, addr ctx.xb[0], addr ctx.k[0], B)
    matmul(L.wv, addr ctx.xb[0], addr ctx.v[0], B)
    for b in 0 ..< B:
      if L.bq.len > 0:
        for i in 0 ..< qDim: ctx.q[b*qDim+i] += L.bq[i]
      if L.bk.len > 0:
        for i in 0 ..< kvDim: ctx.k[b*kvDim+i] += L.bk[i]
      if L.bv.len > 0:
        for i in 0 ..< kvDim: ctx.v[b*kvDim+i] += L.bv[i]
      if L.qNorm.len > 0:
        for h in 0 ..< c.nHeads:
          rmsnorm(ua(ctx.q, b*qDim + h*hd), ua(ctx.q, b*qDim + h*hd), L.qNorm, hd, c.normEps)
      if L.kNorm.len > 0:
        for h in 0 ..< c.nKvHeads:
          rmsnorm(ua(ctx.k, b*kvDim + h*hd), ua(ctx.k, b*kvDim + h*hd), L.kNorm, hd, c.normEps)
      let pos = startPos + b
      rope(ua(ctx.q, b*qDim), c.nHeads, hd, c.ropeDim, pos, m.invFreq, c.ropeNeox)
      rope(ua(ctx.k, b*kvDim), c.nKvHeads, hd, c.ropeDim, pos, m.invFreq, c.ropeNeox)
      let off = (l * ctx.nCtx + pos) * kvDim
      copyMem(addr ctx.kCache[off], addr ctx.k[b*kvDim], kvDim * 4)
      copyMem(addr ctx.vCache[off], addr ctx.v[b*kvDim], kvDim * 4)
    var actx = AttnCtx(q: ua(ctx.q), att: ua(ctx.att), kc: ua(ctx.kCache), vc: ua(ctx.vCache),
                       nHeads: c.nHeads, nKvHeads: c.nKvHeads, hd: hd, nCtx: ctx.nCtx,
                       layer: l, startPos: startPos)
    parallelFor(B * c.nHeads, attnTask, addr actx)
    matmul(L.wo, addr ctx.att[0], addr ctx.xb2[0], B)
    for i in 0 ..< B * c.dim: ctx.x[i] += ctx.xb2[i]
    # FFN SwiGLU (https://dev.to/mshojaei77/swiglu-the-ffn-upgrade-i-use-to-get-free-performance-33jc ❤️)
    for b in 0 ..< B:
      rmsnorm(ua(ctx.xb, b*c.dim), ua(ctx.x, b*c.dim), L.ffnNorm, c.dim, c.normEps)
    matmul(L.wGate, addr ctx.xb[0], addr ctx.hb[0], B)
    matmul(L.wUp, addr ctx.xb[0], addr ctx.hb2[0], B)
    for i in 0 ..< B * c.hidden:
      let g = ctx.hb[i]
      ctx.hb[i] = g / (1'f32 + exp(-g)) * ctx.hb2[i]
    matmul(L.wDown, addr ctx.hb[0], addr ctx.xb2[0], B)
    for i in 0 ..< B * c.dim: ctx.x[i] += ctx.xb2[i]
  for t in toks: ctx.tokens.add t
  # sortie.
  if wantAll:
    for b in 0 ..< B:
      rmsnorm(ua(ctx.xb, b*c.dim), ua(ctx.x, b*c.dim), m.outNorm, c.dim, c.normEps)
    let base = allLogits.len
    allLogits.setLen(base + B * c.vocab)
    matmul(m.output, addr ctx.xb[0], addr allLogits[base], B)
    copyMem(addr ctx.logits[0], addr allLogits[base + (B-1)*c.vocab], c.vocab * 4)
  else:
    rmsnorm(ua(ctx.xb), ua(ctx.x, (B-1)*c.dim), m.outNorm, c.dim, c.normEps)
    matmul(m.output, addr ctx.xb[0], addr ctx.logits[0], 1)
  ctx.lastHidden.setLen(B * c.dim)
  for b in 0 ..< B:
    rmsnorm(ua(ctx.lastHidden, b*c.dim), ua(ctx.x, b*c.dim), m.outNorm, c.dim, c.normEps)

proc eval*(ctx: LlmContext; toks: openArray[int]; allLogits = false): seq[float32] =
  # Ajoute `toks` au contexte et calcule les logits.
  # Retourne les logits du dernier token (ou de tous si `allLogits`, concaténés).
  if toks.len == 0: return ctx.logits
  if ctx.tokens.len + toks.len > ctx.nCtx:
    raise newException(ModelError, "contexte plein (" & $ctx.nCtx & " tokens)")
  let t0 = epochTime()
  var all: seq[float32]
  var i = 0
  while i < toks.len:
    let n = min(ctx.nBatch, toks.len - i)
    ctx.evalBatch(toks[i ..< i+n], allLogits, all)
    i += n
  ctx.evalTime += epochTime() - t0
  ctx.evalTokens += toks.len
  if allLogits: all else: ctx.logits

proc evalPrompt*(ctx: LlmContext; toks: openArray[int]): seq[float32] =
  # Évalue `toks` en réutilisant le préfixe déjà présent dans le cache.
  var common = 0
  while common < min(toks.len, ctx.tokens.len) and toks[common] == ctx.tokens[common]:
    inc common
  if common == toks.len and common > 0:
    # tout est déjà en cache : on réévalue le dernier token pour obtenir ses logits.
    dec common
  ctx.truncate(common)
  ctx.eval(toks[common .. ^1])

proc tokensPerSecond*(ctx: LlmContext): float =
  if ctx.evalTime > 0: ctx.evalTokens.float / ctx.evalTime else: 0

#
# Adaptateurs LoRA (inférence)
#

proc applyLora*(m: LlmModel; path: string) =
  # Charge un adaptateur LoRA (fichier GGUF produit par `nimllm/train`)
  # et l'applique à la volée sans modifier les poids de base.
  let g = openGguf(path)
  let alpha = g.getFloat("adapter.lora.alpha", 16.0)
  proc attach(q: var QMatrix) =
    let an = q.name & ".lora_a"
    let bn = q.name & ".lora_b"
    if an in g.tensors and bn in g.tensors:
      let a = g.tensorF32(an)
      let b = g.tensorF32(bn)
      let rank = g.tensors[an].dims[1]
      q.lora = LoraPair(rank: rank, a: a, b: b, scale: float32(alpha / rank.float))
  for L in m.layers.mitems:
    attach(L.wq); attach(L.wk); attach(L.wv); attach(L.wo)
    attach(L.wGate); attach(L.wUp); attach(L.wDown)
  if not m.cfg.tiedEmbeddings: attach(m.output)
  g.close()

proc removeLora*(m: LlmModel) =
  # Retire tous les adaptateurs LoRA chargés.
  for L in m.layers.mitems:
    for q in [addr L.wq, addr L.wk, addr L.wv, addr L.wo, addr L.wGate, addr L.wUp, addr L.wDown]:
      q[].lora = nil
  m.output.lora = nil

#
# Outils
#

proc embed*(m: LlmModel; text: string; nCtx = 512): seq[float32] =
  # Vecteur de représentation du texte (moyenne des états cachés finaux,
  # normalisée L2). Utile pour la recherche sémantique (RAG).
  let ctx = newContext(m, nCtx = nCtx)
  var toks = m.tokenizer.encode(text, addBos = true, parseSpecial = false)
  if toks.len > ctx.nCtx: toks.setLen(ctx.nCtx)
  let d = m.cfg.dim
  result = newSeq[float32](d)
  var i = 0
  var count = 0
  while i < toks.len:
    let n = min(ctx.nBatch, toks.len - i)
    discard ctx.eval(toks[i ..< i+n])
    for b in 0 ..< n:
      for k in 0 ..< d: result[k] += ctx.lastHidden[b*d + k]
    count += n
    i += n
  var norm = 0.0
  for v in result: norm += float(v) * float(v)
  let inv = float32(1.0 / max(1e-12, sqrt(norm)))
  for v in result.mitems: v *= inv

proc cosineSimilarity*(a, b: openArray[float32]): float =
  var s, na, nb = 0.0
  for i in 0 ..< min(a.len, b.len):
    s += a[i].float * b[i].float
    na += a[i].float * a[i].float
    nb += b[i].float * b[i].float
  s / max(1e-12, sqrt(na) * sqrt(nb))

proc perplexity*(m: LlmModel; text: string; nCtx = 512): float =
  # Perplexité du modèle sur un texte (plus bas = mieux prédit).
  let toks = m.tokenizer.encode(text, addBos = true, parseSpecial = false)
  var nll = 0.0
  var count = 0
  var start = 0
  while start < toks.len - 1:
    let chunk = toks[start ..< min(toks.len, start + nCtx)]
    let ctx = newContext(m, nCtx = nCtx)
    let logits = ctx.eval(chunk, allLogits = true)
    let V = m.cfg.vocab
    for t in 0 ..< chunk.len - 1:
      var mx = -Inf.float32
      for k in 0 ..< V: mx = max(mx, logits[t*V + k])
      var s = 0.0
      for k in 0 ..< V: s += exp(float(logits[t*V + k] - mx))
      nll += -(float(logits[t*V + chunk[t+1]] - mx) - ln(s))
      inc count
    start += nCtx - 1
  exp(nll / max(1, count).float)
