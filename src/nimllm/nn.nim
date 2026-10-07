# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026 (adaptation)
# Portions dérivées de ggml / llama.cpp :
# The ggml authors (© 2023-2026) - LICENSES/MIT-ggml.txt
# nimllm/nn — Transformer entraînable (architecture Llama) et adaptateurs LoRA.
#
# * `newTransformer` : crée un modèle neuf initialisé aléatoirement ;
# * `loadForTraining` : charge un GGUF existant, soit entièrement entraînable
#   (`tmFull`, poids convertis en float32), soit gelé avec adaptateurs LoRA
#   (`tmLora`, poids de base laissés quantifiés et mappés en mémoire) ;
# * `saveGguf`, `saveLora`, `mergeLora` : export vers des fichiers GGUF
#   utilisables par nimllm *et* par llama.cpp / Ollama / LM Studio.

import std/[math, random, tables]
import autograd, model, gguf, quant, tokenizer, parallel, sampler

type
  Linear* = ref object
    # Couche linéaire : y = x·Wˆt (+ LoRA : scale · (x·Aˆt)·Bˆt)
    name*: string                # nom du tenseur GGUF (ex. "blk.0.attn_q.weight").
    inF*, outF*: int
    w*: Tensor                   # poids denses [outF, inF] (nil si gelé quantifié).
    q*: QMatrix                  # poids quantifiés gelés.
    loraA*, loraB*: Tensor       # [r, inF] et [outF, r]
    loraScale*: float32

  Block* = object
    attnNorm*, ffnNorm*: Tensor
    wq*, wk*, wv*, wo*, wGate*, wUp*, wDown*: Linear
    bq*, bk*, bv*: Tensor
    qNorm*, kNorm*: Tensor

  TrainMode* = enum
    tmFull = "complet"           # tous les poids entraînables (float32).
    tmLora = "lora"              # poids de base gelés + adaptateurs de rang faible.

  LoraConfig* = object
    rank*: int                   # rang r (4..64)
    alpha*: float                # facteur d'échelle (scale = alpha / r)
    targets*: seq[string]        # couches visées : "q","k","v","o","gate","up","down","output"

  Transformer* = ref object
    # Modèle de langage entraînable.
    cfg*: ModelConfig
    tokenizer*: Tokenizer
    tokEmb*: Tensor              # [vocab, dim] (nil si gelé quantifié).
    tokEmbQ*: QMatrix
    outNorm*: Tensor
    output*: Linear              # nil si poids liés (tied) à tokEmb.
    blocks*: seq[Block]
    invFreq*: seq[float32]
    mode*: TrainMode
    lora*: LoraConfig
    base*: LlmModel              # modèle de base (mode LoRA ou chargement GGUF).
    rng*: Rand

proc defaultLora*(): LoraConfig =
  # r = 8, alpha = 16, sur les projections d'attention q, k, v, o.
  LoraConfig(rank: 8, alpha: 16, targets: @["q", "k", "v", "o"])

#
# Couche linéaire gelée quantifiée (avec rétropropagation vers l'entrée)
#

type QBackCtx = object
  q: ptr QMatrix
  dy, dx: ptr UncheckedArray[float32]
  n: int

var gQScratch: seq[seq[float32]]

proc qBackTask(p: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  # dx[n, :] += Σm dy[n, m] · W[m, :]  pour n dans [premier, dernier)
  {.cast(gcsafe).}:
    let c = cast[ptr QBackCtx](p)
    let K = c.q.cols
    let M = c.q.rows
    let row = addr gQScratch[worker][0]
    let r = cast[ptr UncheckedArray[float32]](row)
    for m in 0 ..< M:
      var any = false
      for n in first ..< last:
        if c.dy[n*M + m] != 0: any = true; break
      if not any: continue
      dequantRow(c.q.typ, c.q.rowPtr(m), row, K)
      for n in first ..< last:
        let g = c.dy[n*M + m]
        if g == 0: continue
        let dxr = cast[ptr UncheckedArray[float32]](addr c.dx[n*K])
        for k in 0 ..< K: dxr[k] += g * r[k]

proc qlinear*(x: Tensor; q: QMatrix): Tensor =
  # y = x·Wˆt avec W quantifiée et gelée (gradient propagé vers x seulement).
  let N = x.rows
  var shape = x.shape
  shape[^1] = q.rows
  result = newTensor(shape)
  matmul(q, addr x.data[0], addr result.data[0], N)
  if needsGrad(x):
    let r = result
    let qq = q
    result.makeNode(@[x], proc () =
      x.ensureGrad()
      let nt = numThreads()
      if gQScratch.len < nt: gQScratch.setLen(nt)
      for s in gQScratch.mitems:
        if s.len < qq.cols: s.setLen(qq.cols)
      var c = QBackCtx(q: unsafeAddr qq, dy: cast[ptr UncheckedArray[float32]](addr r.grad[0]),
                       dx: cast[ptr UncheckedArray[float32]](addr x.grad[0]), n: N)
      parallelFor(N, qBackTask, addr c, minChunk = max(1, N div nt)))

proc qembedding*(q: QMatrix; ids: openArray[int]): Tensor =
  # Recherche dans une table d'embeddings quantifiée (gelée).
  result = newTensor([ids.len, q.cols])
  for i, id in ids:
    dequantRow(q.typ, q.rowPtr(id), addr result.data[i*q.cols], q.cols)

proc forward*(l: Linear; x: Tensor): Tensor =
  # Applique la couche (poids de base + adaptateur LoRA éventuel).
  result = if l.w != nil: linear(x, l.w) else: qlinear(x, l.q)
  if l.loraA != nil:
    result = result + linear(linear(x, l.loraA), l.loraB).scale(l.loraScale)

proc newLinear(name: string; inF, outF: int; std: float; rng: var Rand): Linear =
  Linear(name: name, inF: inF, outF: outF, w: param([outF, inF], std, rng, name))

proc addLora*(l: Linear; rank: int; alpha: float; rng: var Rand) =
  # Ajoute un adaptateur LoRA (A aléatoire, B nul : le modèle est inchangé au départ).
  l.loraA = param([rank, l.inF], 1.0 / sqrt(l.inF.float), rng, l.name & ".lora_a")
  l.loraB = newTensor([l.outF, rank], true, l.name & ".lora_b")
  l.loraScale = float32(alpha / rank.float)
  if l.w != nil: l.w.requiresGrad = false

#
# Construction
#

proc computeInvFreq(cfg: ModelConfig): seq[float32] =
  let half = cfg.ropeDim div 2
  for i in 0 ..< half:
    result.add float32(pow(cfg.ropeBase, -2.0 * i.float / cfg.ropeDim.float))

proc newModelConfig*(vocab: int; dim = 256; layers = 4; heads = 4; kvHeads = 0;
                     hidden = 0; ctx = 256; ropeBase = 10000.0; tied = true;
                     name = "nimllm-mini"): ModelConfig =
  # Configuration d'une architecture Llama. `hidden` = 0 : ≈ 8/3 · dim
  # arrondi au multiple de 32 ; `kvHeads` = 0 : égal à `heads`.
  result = ModelConfig(arch: "llama", name: name, dim: dim, nLayers: layers,
    nHeads: heads, nKvHeads: (if kvHeads > 0: kvHeads else: heads),
    headDim: dim div heads, vocab: vocab, ctxTrain: ctx, ropeBase: ropeBase,
    ropeDim: dim div heads, normEps: 1e-5, tiedEmbeddings: tied)
  result.hidden = if hidden > 0: hidden else: (dim * 8 div 3 + 31) div 32 * 32
  assert dim mod heads == 0, "dim doit être divisible par heads"
  assert result.nHeads mod result.nKvHeads == 0, "heads doit être multiple de kvHeads"

proc newTransformer*(cfg: ModelConfig; tok: Tokenizer; seed = 42): Transformer =
  # Crée un modèle neuf (initialisation type GPT-2 : N(0, 0.02)).
  result = Transformer(cfg: cfg, tokenizer: tok, mode: tmFull, rng: initRand(seed))
  var rng = initRand(seed)
  let c = cfg
  let std = 0.02
  let projStd = 0.02 / sqrt(2.0 * c.nLayers.float)
  let qd = c.nHeads * c.headDim
  let kvd = c.nKvHeads * c.headDim
  result.tokEmb = param([c.vocab, c.dim], std, rng, "token_embd.weight")
  result.outNorm = full([c.dim], 1, true, "output_norm.weight")
  if not c.tiedEmbeddings:
    result.output = newLinear("output.weight", c.dim, c.vocab, std, rng)
  for l in 0 ..< c.nLayers:
    let p = "blk." & $l & "."
    var b: Block
    b.attnNorm = full([c.dim], 1, true, p & "attn_norm.weight")
    b.ffnNorm = full([c.dim], 1, true, p & "ffn_norm.weight")
    b.wq = newLinear(p & "attn_q.weight", c.dim, qd, std, rng)
    b.wk = newLinear(p & "attn_k.weight", c.dim, kvd, std, rng)
    b.wv = newLinear(p & "attn_v.weight", c.dim, kvd, std, rng)
    b.wo = newLinear(p & "attn_output.weight", qd, c.dim, projStd, rng)
    b.wGate = newLinear(p & "ffn_gate.weight", c.dim, c.hidden, std, rng)
    b.wUp = newLinear(p & "ffn_up.weight", c.dim, c.hidden, std, rng)
    b.wDown = newLinear(p & "ffn_down.weight", c.hidden, c.dim, projStd, rng)
    result.blocks.add b
  result.invFreq = computeInvFreq(cfg)

proc tensorFrom(m: LlmModel; name: string; trainable: bool): Tensor =
  let g = m.gguf
  if name notin g.tensors: return nil
  let ti = g.tensors[name]
  let shape = if ti.dims.len == 1: @[ti.dims[0]] else: @[ti.numElements div ti.dims[0], ti.dims[0]]
  fromSeq(g.tensorF32(name), shape, trainable, name)

proc linearFrom(m: LlmModel; q: QMatrix; trainable: bool): Linear =
  result = Linear(name: q.name, inF: q.cols, outF: q.rows)
  if trainable:
    result.w = tensorFrom(m, q.name, true)
  else:
    result.q = q

proc loadForTraining*(path: string; mode = tmLora; lora = defaultLora();
                      seed = 42; threads = 0): Transformer =
  # Charge un modèle GGUF pour l'entraîner.
  #
  # * `tmFull` : tous les poids deviennent des float32 entraînables
  #   (mémoire ≈ 16 octets/paramètre avec AdamW : réservé aux petits modèles) ;
  # * `tmLora` : poids gelés (restent quantifiés), seuls les adaptateurs
  #   LoRA sont appris (mémoire faible : adapté à llama3.2 1B/3B).
  let m = loadModel(path, threads)
  let c = m.cfg
  result = Transformer(cfg: c, tokenizer: m.tokenizer, mode: mode, lora: lora,
                       base: m, rng: initRand(seed), invFreq: m.invFreq)
  let full = mode == tmFull
  if full:
    result.tokEmb = tensorFrom(m, "token_embd.weight", true)
  else:
    result.tokEmbQ = m.tokEmb
  result.outNorm = tensorFrom(m, "output_norm.weight", full)
  if not c.tiedEmbeddings:
    result.output = linearFrom(m, m.output, full)
  elif not full:
    # tête de sortie liée et gelée : on passe par la matrice quantifiée
    result.output = Linear(name: "token_embd.weight", inF: c.dim, outF: c.vocab, q: m.tokEmb)
  for l in 0 ..< c.nLayers:
    let L = m.layers[l]
    let p = "blk." & $l & "."
    var b: Block
    b.attnNorm = tensorFrom(m, p & "attn_norm.weight", full)
    b.ffnNorm = tensorFrom(m, p & "ffn_norm.weight", full)
    b.wq = linearFrom(m, L.wq, full); b.wk = linearFrom(m, L.wk, full)
    b.wv = linearFrom(m, L.wv, full); b.wo = linearFrom(m, L.wo, full)
    b.wGate = linearFrom(m, L.wGate, full); b.wUp = linearFrom(m, L.wUp, full)
    b.wDown = linearFrom(m, L.wDown, full)
    b.bq = tensorFrom(m, p & "attn_q.bias", full)
    b.bk = tensorFrom(m, p & "attn_k.bias", full)
    b.bv = tensorFrom(m, p & "attn_v.bias", full)
    b.qNorm = tensorFrom(m, p & "attn_q_norm.weight", full)
    b.kNorm = tensorFrom(m, p & "attn_k_norm.weight", full)
    if not full:
      var rng = result.rng
      for (key, lin) in [("q", b.wq), ("k", b.wk), ("v", b.wv), ("o", b.wo),
                         ("gate", b.wGate), ("up", b.wUp), ("down", b.wDown)]:
        if key in lora.targets: lin.addLora(lora.rank, lora.alpha, rng)
      result.rng = rng
    result.blocks.add b
  if not full and "output" in lora.targets and result.output != nil:
    result.output.addLora(lora.rank, lora.alpha, result.rng)

proc parameters*(t: Transformer): seq[Tensor] =
  # Tenseurs entraînables (ceux qui recevront un gradient).
  proc addT(res: var seq[Tensor]; x: Tensor) =
    if x != nil and x.requiresGrad: res.add x
  proc addL(res: var seq[Tensor]; l: Linear) =
    if l == nil: return
    addT(res, l.w); addT(res, l.loraA); addT(res, l.loraB)
  addT(result, t.tokEmb)
  for b in t.blocks:
    addT(result, b.attnNorm)
    for l in [b.wq, b.wk, b.wv, b.wo, b.wGate, b.wUp, b.wDown]: addL(result, l)
    addT(result, b.ffnNorm)
    addT(result, b.bq); addT(result, b.bk); addT(result, b.bv)
    addT(result, b.qNorm); addT(result, b.kNorm)
  addT(result, t.outNorm)
  addL(result, t.output)

proc parameterCount*(t: Transformer; trainableOnly = true): int =
  for p in t.parameters: result += p.numel
  if not trainableOnly and t.base != nil: result = t.base.paramCount

#
# Propagation avant
#

proc headNorm(x, w: Tensor; nHeads, headDim: int; eps: float): Tensor =
  let n = x.rows
  rmsnorm(x.reshape([n * nHeads, headDim]), w, eps).reshape([n, nHeads * headDim])

proc hidden*(t: Transformer; ids: openArray[int]; B, T: int): Tensor =
  # États cachés finaux normalisés [B*T, dim].
  let c = t.cfg
  assert ids.len == B * T
  var x = if t.tokEmb != nil: embedding(t.tokEmb, ids) else: qembedding(t.tokEmbQ, ids)
  var positions = newSeq[int](B*T)
  for i in 0 ..< B*T: positions[i] = i mod T
  for b in t.blocks:
    let h = rmsnorm(x, b.attnNorm, c.normEps)
    var q = b.wq.forward(h)
    var k = b.wk.forward(h)
    var v = b.wv.forward(h)
    if b.bq != nil: q = q + b.bq
    if b.bk != nil: k = k + b.bk
    if b.bv != nil: v = v + b.bv
    if b.qNorm != nil: q = headNorm(q, b.qNorm, c.nHeads, c.headDim, c.normEps)
    if b.kNorm != nil: k = headNorm(k, b.kNorm, c.nKvHeads, c.headDim, c.normEps)
    q = rope(q, c.nHeads, c.headDim, positions, t.invFreq, c.ropeNeox)
    k = rope(k, c.nKvHeads, c.headDim, positions, t.invFreq, c.ropeNeox)
    let a = causalAttention(q, k, v, B, T, c.nHeads, c.nKvHeads, c.headDim)
    x = x + b.wo.forward(a)
    let h2 = rmsnorm(x, b.ffnNorm, c.normEps)
    x = x + b.wDown.forward(silu(b.wGate.forward(h2)) * b.wUp.forward(h2))
  rmsnorm(x, t.outNorm, c.normEps)

proc forward*(t: Transformer; ids: openArray[int]; B, T: int): Tensor =
  # Logits [B*T, vocab] pour un lot de B séquences de T tokens.
  let x = t.hidden(ids, B, T)
  if t.output != nil: t.output.forward(x)
  else: linear(x, t.tokEmb)

proc loss*(t: Transformer; inputs, targets: seq[int]; B, T: int): Tensor =
  # Perte d'entropie croisée (prédiction du token suivant).
  crossEntropy(t.forward(inputs, B, T), targets)

proc generateText*(t: Transformer; prompt: string; maxTokens = 50; temperature = 0.8;
                   seed = 1; addBos = true): string =
  # Génération rapide pour surveiller l'entraînement (recalcule toute la
  # séquence à chaque token : réservé aux petits modèles / textes courts).
  var ids = t.tokenizer.encode(prompt, addBos = addBos and t.tokenizer.bosId >= 0)
  if ids.len == 0: ids = @[max(0, t.tokenizer.bosId)]
  var s = newSampler(SamplerParams(temperature: temperature, topK: 40, topP: 0.95,
                                   repeatPenalty: 1.0, seed: seed))
  var outIds: seq[int]
  noGrad:
    for i in 0 ..< maxTokens:
      let ctxIds = if ids.len > t.cfg.ctxTrain: ids[^t.cfg.ctxTrain .. ^1] else: ids
      let logits = t.forward(ctxIds, 1, ctxIds.len)
      let V = t.cfg.vocab
      let last = logits.data[(ctxIds.len - 1)*V ..< ctxIds.len*V]
      let id = s.sample(last)
      if id in t.tokenizer.eogIds: break
      ids.add id
      outIds.add id
  t.tokenizer.decode(outIds)

#
# Export
#

proc writeArchKeys*(w: GgufWriter; c: ModelConfig) =
  # Métadonnées d'architecture au format llama.cpp.
  let a = c.arch
  w.setKV("general.architecture", gStr(a))
  w.setKV("general.name", gStr(c.name))
  w.setKV(a & ".context_length", gU32(c.ctxTrain))
  w.setKV(a & ".embedding_length", gU32(c.dim))
  w.setKV(a & ".feed_forward_length", gU32(c.hidden))
  w.setKV(a & ".block_count", gU32(c.nLayers))
  w.setKV(a & ".attention.head_count", gU32(c.nHeads))
  w.setKV(a & ".attention.head_count_kv", gU32(c.nKvHeads))
  w.setKV(a & ".rope.freq_base", gF32(c.ropeBase))
  w.setKV(a & ".rope.dimension_count", gU32(c.ropeDim))
  w.setKV(a & ".attention.layer_norm_rms_epsilon", gF32(c.normEps))
  w.setKV(a & ".vocab_size", gU32(c.vocab))
  if c.headDim * c.nHeads != c.dim:
    w.setKV(a & ".attention.key_length", gU32(c.headDim))
    w.setKV(a & ".attention.value_length", gU32(c.headDim))

proc ggufDims(t: Tensor): seq[int] =
  if t.shape.len == 1: @[t.shape[0]] else: @[t.shape[^1], t.rows]

proc saveGguf*(t: Transformer; path: string; wtype = gtF32; extra: seq[(string, GgufValue)] = @[]) =
  # Exporte le modèle complet en GGUF (architecture llama).
  # `wtype` : type des matrices (F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K...) ;
  # les vecteurs (normes, biais) restent en F32.
  # En mode LoRA, utilisez plutôt `saveLora` puis `mergeLora`.
  if t.mode == tmLora:
    raise newException(ValueError, "modèle LoRA : utilisez saveLora puis mergeLora")
  let w = newGgufWriter()
  writeArchKeys(w, t.cfg)
  w.setKV("general.file_type", gU32(if wtype == gtF32: 0 elif wtype == gtF16: 1 else: 7))
  t.tokenizer.writeToGguf(w)
  for (k, v) in extra: w.setKV(k, v)
  proc put(x: Tensor; typ: GgmlType) =
    if x != nil: w.addTensorF32(x.name, ggufDims(x), x.data, typ)
  # l'embedding est souvent gardé en meilleure précision
  put(t.tokEmb, (if wtype in {gtQ4_0, gtQ4_1, gtQ4_K, gtQ5_K, gtQ5_0, gtQ5_1}: gtQ8_0 else: wtype))
  put(t.outNorm, gtF32)
  if t.output != nil: put(t.output.w, (if wtype in {gtQ4_0, gtQ4_K}: gtQ6_K else: wtype))
  for b in t.blocks:
    put(b.attnNorm, gtF32); put(b.ffnNorm, gtF32)
    for l in [b.wq, b.wk, b.wv, b.wo, b.wGate, b.wUp, b.wDown]: put(l.w, wtype)
    put(b.bq, gtF32); put(b.bk, gtF32); put(b.bv, gtF32)
    put(b.qNorm, gtF32); put(b.kNorm, gtF32)
  w.write(path)

proc saveLora*(t: Transformer; path: string) =
  # Enregistre uniquement les adaptateurs LoRA (fichier GGUF léger).
  let w = newGgufWriter()
  w.setKV("general.architecture", gStr(t.cfg.arch))
  w.setKV("general.type", gStr("adapter"))
  w.setKV("adapter.type", gStr("lora"))
  w.setKV("adapter.lora.alpha", gF32(t.lora.alpha))
  proc put(l: Linear) =
    if l == nil or l.loraA == nil: return
    w.addTensorF32(l.name & ".lora_a", @[l.inF, l.loraA.shape[0]], l.loraA.data)
    w.addTensorF32(l.name & ".lora_b", @[l.loraB.shape[1], l.outF], l.loraB.data)
  for b in t.blocks:
    for l in [b.wq, b.wk, b.wv, b.wo, b.wGate, b.wUp, b.wDown]: put(l)
  put(t.output)
  w.write(path)

proc mergeLora*(basePath, adapterPath, outPath: string; outType = gtQ8_0) =
  # Fusionne un adaptateur LoRA dans le modèle de base : W' = W + scale·B·A,
  # et écrit un nouveau GGUF autonome. Les tenseurs non adaptés sont copiés
  # tels quels ; les tenseurs modifiés sont re-quantifiés en `outType`.
  let base = openGguf(basePath)
  let ad = openGguf(adapterPath)
  let alpha = ad.getFloat("adapter.lora.alpha", 16)
  let w = newGgufWriter()
  w.copyMetadata(base)
  var keep: seq[seq[float32]]
  for name, ti in base.tensors:
    let an = name & ".lora_a"
    let bn = name & ".lora_b"
    if an in ad.tensors and bn in ad.tensors:
      var W = base.tensorF32(name)
      let A = ad.tensorF32(an)
      let Bm = ad.tensorF32(bn)
      let K = ti.dims[0]
      let M = W.len div K
      let r = ad.tensors[an].dims[1]
      let s = float32(alpha / r.float)
      for m in 0 ..< M:
        for i in 0 ..< r:
          let bv = Bm[m*r + i] * s
          if bv == 0: continue
          for k in 0 ..< K: W[m*K + k] += bv * A[i*K + k]
      let typ = if outType in quantizableTypes and K mod blockSize(outType) == 0: outType else: gtF16
      w.addTensorF32(name, ti.dims, W, typ)
    else:
      w.addTensorRaw(name, ti.dims, ti.typ, ti.data, ti.byteSize)
  w.write(outPath)
  base.close(); ad.close()

proc quantizeModel*(inPath, outPath: string; wtype: GgmlType; keepOutput = true) =
  # Re-quantifie un modèle GGUF (ex. F32 -> Q4_K). Les vecteurs restent F32 ;
  # l'embedding et la tête de sortie sont gardés en Q8_0/Q6_K si `keepOutput`.
  let g = openGguf(inPath)
  let w = newGgufWriter()
  w.copyMetadata(g)
  for name, ti in g.tensors:
    if ti.dims.len < 2:
      w.addTensorRaw(name, ti.dims, ti.typ, ti.data, ti.byteSize)
      continue
    var t = wtype
    if keepOutput and (name == "token_embd.weight" or name == "output.weight"):
      t = if wtype in {gtF32, gtF16, gtBF16, gtQ8_0}: wtype else: gtQ8_0
    if ti.dims[0] mod blockSize(t) != 0: t = gtF16
    w.addTensorF32(name, ti.dims, g.tensorF32(name), t)
  w.write(outPath)
  g.close()
