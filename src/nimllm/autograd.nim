# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/autograd — tenseurs float32 avec différentiation automatique
# (mode inverse, « backpropagation »).
#
# Chaque opération crée un nouveau `Tensor` qui mémorise ses parents et une
# fonction de rétropropagation. `backward(loss)` parcourt le graphe en ordre
# topologique inverse et accumule les gradients dans `t.grad`.
#
# Convention : les matrices sont stockées ligne par ligne ; un tenseur de
# forme [N, C] contient N lignes de C valeurs.

import std/[math, random, sets]
import parallel, quant

type
  Tensor* = ref object
    data*: seq[float32]
    grad*: seq[float32]
    shape*: seq[int]
    requiresGrad*: bool
    name*: string
    parents: seq[Tensor]
    backFn: proc () {.closure.}

var gradEnabled* = true
  # Mis à false par `noGrad` : aucun graphe n'est construit (inférence, évaluation).

template noGrad*(body: untyped) =
  ## Exécute `body` sans enregistrer de graphe de calcul.
  let old = gradEnabled
  gradEnabled = false
  try: body
  finally: gradEnabled = old

#
# Construction
#

proc numel*(shape: openArray[int]): int =
  result = 1
  for d in shape: result *= d

proc numel*(t: Tensor): int = t.data.len
proc cols*(t: Tensor): int = t.shape[^1]
proc rows*(t: Tensor): int = t.data.len div max(1, t.shape[^1])

proc `$`*(t: Tensor): string =
  result = "Tensor" & $t.shape
  if t.name.len > 0: result.add "(" & t.name & ")"
  if t.data.len <= 8: result.add " " & $t.data
  else: result.add " [" & $t.data[0] & ", " & $t.data[1] & ", ... " & $t.data[^1] & "]"

proc newTensor*(shape: openArray[int]; requiresGrad = false; name = ""): Tensor =
  ## Tenseur rempli de zéros.
  Tensor(data: newSeq[float32](numel(shape)), shape: @shape, requiresGrad: requiresGrad, name: name)

proc fromSeq*(data: seq[float32]; shape: openArray[int]; requiresGrad = false; name = ""): Tensor =
  assert data.len == numel(shape), "taille incompatible avec la forme"
  Tensor(data: data, shape: @shape, requiresGrad: requiresGrad, name: name)

proc scalar*(v: float32): Tensor = fromSeq(@[v], [1])

proc randn*(shape: openArray[int]; std = 1.0; rng: var Rand; requiresGrad = true; name = ""): Tensor =
  # Initialisation gaussienne N(0, std²).
  result = newTensor(shape, requiresGrad, name)
  for v in result.data.mitems: v = float32(rng.gauss() * std)

proc full*(shape: openArray[int]; value: float32; requiresGrad = false; name = ""): Tensor =
  result = newTensor(shape, requiresGrad, name)
  for v in result.data.mitems: v = value

proc param*(shape: openArray[int]; std: float; rng: var Rand; name = ""): Tensor =
  # Paramètre entraînable initialisé aléatoirement.
  randn(shape, std, rng, true, name)

proc item*(t: Tensor): float32 = t.data[0]

proc ensureGrad*(t: Tensor) {.inline.} =
  if t.grad.len != t.data.len: t.grad = newSeq[float32](t.data.len)

proc zeroGrad*(t: Tensor) =
  if t.grad.len > 0:
    for g in t.grad.mitems: g = 0

proc detach*(t: Tensor): Tensor =
  # Copie sans historique de calcul.
  Tensor(data: t.data, shape: t.shape)

proc reshape*(t: Tensor; shape: openArray[int]): Tensor =
  # Nouvelle vue (copie des données) avec une autre forme.
  assert numel(shape) == t.data.len
  result = Tensor(data: t.data, shape: @shape, requiresGrad: t.requiresGrad and gradEnabled)
  if result.requiresGrad:
    let r = result
    result.parents = @[t]
    result.backFn = proc () =
      t.ensureGrad()
      for i in 0 ..< r.grad.len: t.grad[i] += r.grad[i]

proc needGrad(ts: varargs[Tensor]): bool =
  if not gradEnabled: return false
  for t in ts:
    if t != nil and t.requiresGrad: return true

proc node(res: Tensor; parents: seq[Tensor]; fn: proc () {.closure.}) =
  res.requiresGrad = true
  res.parents = parents
  res.backFn = fn

proc needsGrad*(ts: varargs[Tensor]): bool =
  # Vrai si un graphe doit être construit pour ces entrées.
  if not gradEnabled: return false
  for t in ts:
    if t != nil and t.requiresGrad: return true

proc makeNode*(res: Tensor; parents: seq[Tensor]; backward: proc () {.closure.}) =
  # Déclare `res` comme résultat d'une opération personnalisée : `backward`
  # doit lire `res.grad` et accumuler dans `parent.grad` (après `ensureGrad`).
  node(res, parents, backward)

#
# Rétropropagation
#

proc backward*(loss: Tensor) =
  # Calcule les gradients de `loss` (scalaire) par rapport à tous les
  # tenseurs dont il dépend et qui ont `requiresGrad`.
  var order: seq[Tensor]
  var seen = initHashSet[pointer]()
  var stack: seq[(Tensor, bool)] = @[(loss, false)]
  while stack.len > 0:
    let (t, done) = stack.pop()
    if done:
      order.add t; continue
    if cast[pointer](t) in seen: continue
    seen.incl cast[pointer](t)
    stack.add (t, true)
    for p in t.parents:
      if p.requiresGrad and cast[pointer](p) notin seen: stack.add (p, false)
  loss.ensureGrad()
  for i in 0 ..< loss.grad.len: loss.grad[i] = 1
  for i in countdown(order.high, 0):
    let t = order[i]
    if t.backFn != nil and t.grad.len > 0:
      t.backFn()
    # un résultat intermédiaire n'est plus utile une fois son gradient propagé :
    # on libère sa mémoire (sauf la perte elle-même et les feuilles).
    if t.backFn != nil and t != loss:
      t.grad = @[]
      t.data = @[]
  # libère le graphe (les gradients des feuilles restent)
  for t in order:
    if t.backFn != nil:
      t.backFn = nil
      t.parents = @[]

#
# Noyaux parallèles
#

{.push checks: off, boundChecks: off, overflowChecks: off.}

type
  F = ptr UncheckedArray[float32]
  LinCtx = object
    x, w, y: F
    n, k, m: int
    accumulate: bool

template fp(s: seq[float32]): F = cast[F](unsafeAddr s[0])

proc linFwdTask(p: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  # y[n, m] = Σk x[n, k] * w[m, k] — par blocs de 4 lignes de x.
  let c = cast[ptr LinCtx](p)
  let K = c.k
  let M = c.m
  var nb = first * 4
  let nEnd = min(c.n, last * 4)
  while nb < nEnd:
    let cnt = min(4, nEnd - nb)
    if cnt == 4:
      let x0 = cast[F](addr c.x[nb*K])
      let x1 = cast[F](addr c.x[(nb+1)*K])
      let x2 = cast[F](addr c.x[(nb+2)*K])
      let x3 = cast[F](addr c.x[(nb+3)*K])
      for m in 0 ..< M:
        let w = cast[F](addr c.w[m*K])
        var s0, s1, s2, s3: float32
        for k in 0 ..< K:
          let wv = w[k]
          s0 += x0[k]*wv; s1 += x1[k]*wv; s2 += x2[k]*wv; s3 += x3[k]*wv
        if c.accumulate:
          c.y[nb*M + m] += s0; c.y[(nb+1)*M + m] += s1
          c.y[(nb+2)*M + m] += s2; c.y[(nb+3)*M + m] += s3
        else:
          c.y[nb*M + m] = s0; c.y[(nb+1)*M + m] = s1
          c.y[(nb+2)*M + m] = s2; c.y[(nb+3)*M + m] = s3
    else:
      for r in nb ..< nb + cnt:
        for m in 0 ..< M:
          let s = dotF32(addr c.x[r*K], addr c.w[m*K], K)
          if c.accumulate: c.y[r*M + m] += s else: c.y[r*M + m] = s
    nb += 4

proc linearRaw*(x: F; w: F; y: F; n, k, m: int; accumulate = false) =
  # y[n, m] (+)= x[n, k] · w[m, k]ᵀ
  var c = LinCtx(x: x, w: w, y: y, n: n, k: k, m: m, accumulate: accumulate)
  parallelFor((n + 3) div 4, linFwdTask, addr c)

type AxpyCtx = object
  a, b, out0: F                  # out[i, :] += Σj a[i, j] * b[j, :] ou variante transposée.
  ni, nj, nk: int
  transA: bool

proc gemmTask(p: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  let c = cast[ptr AxpyCtx](p)
  let K = c.nk
  for i in first ..< last:
    let o = cast[F](addr c.out0[i*K])
    for j in 0 ..< c.nj:
      let s = if c.transA: c.a[j*c.ni + i] else: c.a[i*c.nj + j]
      if s == 0: continue
      let bj = cast[F](addr c.b[j*K])
      for k in 0 ..< K: o[k] += s * bj[k]

proc gemmAcc*(a, b, outp: F; ni, nj, nk: int; transA: bool) =
  # out[ni, nk] += A[ni, nj] · B[nj, nk]   (A transposée si `transA` : A est [nj, ni]).
  var c = AxpyCtx(a: a, b: b, out0: outp, ni: ni, nj: nj, nk: nk, transA: transA)
  parallelFor(ni, gemmTask, addr c)

{.pop.}

#
# Opérations
#

proc linear*(x, w: Tensor; bias: Tensor = nil): Tensor =
  # y = x · wᵀ (+ b).  x : [N, K], w : [M, K]  ->  y : [N, M]
  let K = x.cols
  let N = x.rows
  let M = w.shape[0]
  assert w.cols == K, "linear : dimensions incompatibles " & $x.shape & " · " & $w.shape & "ᵀ"
  var shape = x.shape
  shape[^1] = M
  result = newTensor(shape)
  linearRaw(fp(x.data), fp(w.data), fp(result.data), N, K, M)
  if bias != nil:
    for n in 0 ..< N:
      for m in 0 ..< M: result.data[n*M + m] += bias.data[m]
  if needGrad(x, w, bias):
    let r = result
    result.node(@[x, w] & (if bias != nil: @[bias] else: @[]), proc () =
      if x.requiresGrad:
        x.ensureGrad()
        gemmAcc(fp(r.grad), fp(w.data), fp(x.grad), N, M, K, false)   # dx = dy · w
      if w.requiresGrad:
        w.ensureGrad()
        gemmAcc(fp(r.grad), fp(x.data), fp(w.grad), M, N, K, true)    # dw = dyᵀ · x
      if bias != nil and bias.requiresGrad:
        bias.ensureGrad()
        for n in 0 ..< N:
          for m in 0 ..< M: bias.grad[m] += r.grad[n*M + m])

proc matmul*(a, b: Tensor): Tensor =
  # Produit matriciel classique : a [N, K] · b [K, M] -> [N, M].
  let N = a.rows
  let K = a.cols
  let M = b.cols
  assert b.rows == K
  result = newTensor([N, M])
  gemmAcc(fp(a.data), fp(b.data), fp(result.data), N, K, M, false)
  if needGrad(a, b):
    let r = result
    result.node(@[a, b], proc () =
      if a.requiresGrad:
        a.ensureGrad()
        # da = dy · bᵀ  : linéaire (dy [N,M], b [K,M]) -> [N,K]
        linearRaw(fp(r.grad), fp(b.data), fp(a.grad), N, M, K, accumulate = true)
      if b.requiresGrad:
        b.ensureGrad()
        gemmAcc(fp(a.data), fp(r.grad), fp(b.grad), K, N, M, true))    # db = aᵀ · dy

proc binaryOp(a, b: Tensor; op: char): Tensor =
  # b peut avoir la même forme que a, ou être un vecteur ligne diffusé [C].
  let broadcast = b.data.len != a.data.len
  if broadcast: assert b.data.len == a.cols, "diffusion impossible " & $a.shape & " / " & $b.shape
  result = newTensor(a.shape)
  let C = b.data.len
  for i in 0 ..< a.data.len:
    let bv = if broadcast: b.data[i mod C] else: b.data[i]
    result.data[i] = case op
      of '+': a.data[i] + bv
      of '-': a.data[i] - bv
      else: a.data[i] * bv
  if needGrad(a, b):
    let r = result
    result.node(@[a, b], proc () =
      if a.requiresGrad:
        a.ensureGrad()
        for i in 0 ..< r.grad.len:
          let bv = if broadcast: b.data[i mod C] else: b.data[i]
          a.grad[i] += (if op == '*': r.grad[i] * bv else: r.grad[i])
      if b.requiresGrad:
        b.ensureGrad()
        for i in 0 ..< r.grad.len:
          let j = if broadcast: i mod C else: i
          b.grad[j] += (case op
            of '+': r.grad[i]
            of '-': -r.grad[i]
            else: r.grad[i] * a.data[i]))

proc `+`*(a, b: Tensor): Tensor = binaryOp(a, b, '+')
proc `-`*(a, b: Tensor): Tensor = binaryOp(a, b, '-')
proc `*`*(a, b: Tensor): Tensor = binaryOp(a, b, '*')

proc scale*(a: Tensor; s: float32): Tensor =
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len: result.data[i] = a.data[i] * s
  if needGrad(a):
    let r = result
    result.node(@[a], proc () =
      a.ensureGrad()
      for i in 0 ..< r.grad.len: a.grad[i] += r.grad[i] * s)

proc `*`*(a: Tensor; s: float32): Tensor = scale(a, s)

proc unary(a: Tensor; f: proc (x: float32): float32; df: proc (x, y: float32): float32): Tensor =
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len: result.data[i] = f(a.data[i])
  if needGrad(a):
    let r = result
    result.node(@[a], proc () =
      a.ensureGrad()
      for i in 0 ..< r.grad.len: a.grad[i] += r.grad[i] * df(a.data[i], r.data[i]))

proc relu*(a: Tensor): Tensor =
  unary(a, proc (x: float32): float32 = max(x, 0),
           proc (x, y: float32): float32 = (if x > 0: 1 else: 0))

proc silu*(a: Tensor): Tensor =
  # x * sigmoïde(x)
  unary(a, proc (x: float32): float32 = x / (1 + exp(-x)),
           proc (x, y: float32): float32 =
             let s = 1 / (1 + exp(-x))
             s * (1 + x * (1 - s)))

proc sigmoid*(a: Tensor): Tensor =
  unary(a, proc (x: float32): float32 = 1 / (1 + exp(-x)),
           proc (x, y: float32): float32 = y * (1 - y))

proc tanhT*(a: Tensor): Tensor =
  unary(a, proc (x: float32): float32 = tanh(x),
           proc (x, y: float32): float32 = 1 - y*y)

proc gelu*(a: Tensor): Tensor =
  # GELU (approximation tanh).
  const c = 0.7978845608'f32
  unary(a, proc (x: float32): float32 = 0.5'f32 * x * (1 + tanh(c * (x + 0.044715'f32*x*x*x))),
           proc (x, y: float32): float32 =
             let t = tanh(c * (x + 0.044715'f32*x*x*x))
             0.5'f32 * (1 + t) + 0.5'f32 * x * (1 - t*t) * c * (1 + 3*0.044715'f32*x*x))

proc sum*(a: Tensor): Tensor =
  var s = 0.0
  for v in a.data: s += v
  result = scalar(float32(s))
  if needGrad(a):
    let r = result
    result.node(@[a], proc () =
      a.ensureGrad()
      for i in 0 ..< a.grad.len: a.grad[i] += r.grad[0])

proc mean*(a: Tensor): Tensor = sum(a).scale(1'f32 / a.data.len.float32)

proc mseLoss*(pred, target: Tensor): Tensor =
  # Erreur quadratique moyenne.
  let d = pred - target
  mean(d * d)

proc softmax*(a: Tensor): Tensor =
  # Softmax par ligne.
  let C = a.cols
  let N = a.rows
  result = newTensor(a.shape)
  for n in 0 ..< N:
    var mx = -Inf.float32
    for c in 0 ..< C: mx = max(mx, a.data[n*C + c])
    var s = 0'f32
    for c in 0 ..< C:
      let e = exp(a.data[n*C + c] - mx)
      result.data[n*C + c] = e; s += e
    for c in 0 ..< C: result.data[n*C + c] /= s
  if needGrad(a):
    let r = result
    result.node(@[a], proc () =
      a.ensureGrad()
      for n in 0 ..< N:
        var dot = 0'f32
        for c in 0 ..< C: dot += r.grad[n*C + c] * r.data[n*C + c]
        for c in 0 ..< C:
          a.grad[n*C + c] += r.data[n*C + c] * (r.grad[n*C + c] - dot))

proc embedding*(table: Tensor; ids: openArray[int]): Tensor =
  # Sélectionne les lignes `ids` de `table` [V, C] -> [len(ids), C].
  let C = table.cols
  result = newTensor([ids.len, C])
  for i, id in ids:
    copyMem(addr result.data[i*C], addr table.data[id*C], C*4)
  if needGrad(table):
    let r = result
    let idsCopy = @ids
    result.node(@[table], proc () =
      table.ensureGrad()
      for i, id in idsCopy:
        for c in 0 ..< C: table.grad[id*C + c] += r.grad[i*C + c])

proc rmsnorm*(x, w: Tensor; eps = 1e-5): Tensor =
  # Normalisation RMS par ligne, puis multiplication par le gain `w` [C].
  let C = x.cols
  let N = x.rows
  result = newTensor(x.shape)
  var inv = newSeq[float32](N)
  for n in 0 ..< N:
    var ss = 0.0
    for c in 0 ..< C: ss += float(x.data[n*C + c])^2
    inv[n] = float32(1.0 / sqrt(ss / C.float + eps))
    for c in 0 ..< C: result.data[n*C + c] = x.data[n*C + c] * inv[n] * w.data[c]
  if needGrad(x, w):
    let r = result
    result.node(@[x, w], proc () =
      if w.requiresGrad: w.ensureGrad()
      if x.requiresGrad: x.ensureGrad()
      for n in 0 ..< N:
        let s = inv[n]
        var dot = 0'f32   # Σ dy·w·x
        for c in 0 ..< C:
          let g = r.grad[n*C + c]
          dot += g * w.data[c] * x.data[n*C + c]
          if w.requiresGrad: w.grad[c] += g * x.data[n*C + c] * s
        if x.requiresGrad:
          let k = dot * s * s * s / C.float32
          for c in 0 ..< C:
            x.grad[n*C + c] += r.grad[n*C + c] * w.data[c] * s - x.data[n*C + c] * k)

proc ropeApply(d: var seq[float32]; base, nHeads, headDim: int; pos: int;
               invFreq: seq[float32]; neox: bool; sign: float32) =
  let half = invFreq.len
  for h in 0 ..< nHeads:
    let o = base + h*headDim
    for i in 0 ..< half:
      let th = pos.float32 * invFreq[i]
      let c = cos(th)
      let s = sin(th) * sign
      let (a, b) = if neox: (o + i, o + i + half) else: (o + 2*i, o + 2*i + 1)
      let x0 = d[a]
      let x1 = d[b]
      d[a] = x0*c - x1*s
      d[b] = x0*s + x1*c

proc rope*(x: Tensor; nHeads, headDim: int; positions: seq[int];
           invFreq: seq[float32]; neox = false): Tensor =
  # Encodage positionnel rotatif. x : [N, nHeads*headDim], positions : N valeurs.
  let C = x.cols
  result = Tensor(data: x.data, shape: x.shape)
  for n in 0 ..< x.rows:
    ropeApply(result.data, n*C, nHeads, headDim, positions[n], invFreq, neox, 1)
  if needGrad(x):
    let r = result
    result.node(@[x], proc () =
      x.ensureGrad()
      var g = r.grad
      for n in 0 ..< x.rows:
        ropeApply(g, n*C, nHeads, headDim, positions[n], invFreq, neox, -1)
      for i in 0 ..< g.len: x.grad[i] += g[i])

{.push checks: off, boundChecks: off, overflowChecks: off.}
type AttCtx = object
  q, k, v, o, p: F               # p : probabilités [B, H, T, T]
  dq, dk, dv, gOut: F
  B, T, H, Hkv, D: int
  scale: float32

proc attFwdTask(pp: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  let c = cast[ptr AttCtx](pp)
  let T = c.T
  let D = c.D
  let qd = c.H * D
  let kd = c.Hkv * D
  let g = c.H div c.Hkv
  for job in first ..< last:     # job = (b, h, t)
    let b = job div (c.H * T)
    let h = (job div T) mod c.H
    let t = job mod T
    let kh = h div g
    let qrow = cast[F](addr c.q[(b*T + t)*qd + h*D])
    let prow = cast[F](addr c.p[((b*c.H + h)*T + t)*T])
    var mx = -Inf.float32
    for j in 0 .. t:
      let s = dotF32(addr qrow[0], addr c.k[(b*T + j)*kd + kh*D], D) * c.scale
      prow[j] = s
      mx = max(mx, s)
    var sum = 0'f32
    for j in 0 .. t:
      prow[j] = exp(prow[j] - mx); sum += prow[j]
    for j in 0 .. t: prow[j] /= sum
    for j in t+1 ..< T: prow[j] = 0
    let orow = cast[F](addr c.o[(b*T + t)*qd + h*D])
    for d in 0 ..< D: orow[d] = 0
    for j in 0 .. t:
      let w = prow[j]
      let vr = cast[F](addr c.v[(b*T + j)*kd + kh*D])
      for d in 0 ..< D: orow[d] += w * vr[d]

proc attBwdTask(pp: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  # Une tâche = (b, groupe kv) : évite les écritures concurrentes sur dk/dv.
  let c = cast[ptr AttCtx](pp)
  let T = c.T
  let D = c.D
  let qd = c.H * D
  let kd = c.Hkv * D
  let g = c.H div c.Hkv
  var dp = newSeq[float32](T)
  for job in first ..< last:
    let b = job div c.Hkv
    let kh = job mod c.Hkv
    for h in kh*g ..< (kh+1)*g:
      for t in 0 ..< T:
        let prow = cast[F](addr c.p[((b*c.H + h)*T + t)*T])
        let dorow = cast[F](addr c.gOut[(b*T + t)*qd + h*D])
        # dP = gOut · Vᵀ ; dV += Pᵀ · gOut
        var dot = 0'f32
        for j in 0 .. t:
          let vr = cast[F](addr c.v[(b*T + j)*kd + kh*D])
          dp[j] = dotF32(addr dorow[0], addr vr[0], D)
          dot += dp[j] * prow[j]
          let dvr = cast[F](addr c.dv[(b*T + j)*kd + kh*D])
          let w = prow[j]
          for d in 0 ..< D: dvr[d] += w * dorow[d]
        # dS = P ⊙ (dP - Σ P·dP)
        let qrow = cast[F](addr c.q[(b*T + t)*qd + h*D])
        let dqrow = cast[F](addr c.dq[(b*T + t)*qd + h*D])
        for j in 0 .. t:
          let ds = prow[j] * (dp[j] - dot) * c.scale
          if ds == 0: continue
          let kr = cast[F](addr c.k[(b*T + j)*kd + kh*D])
          let dkr = cast[F](addr c.dk[(b*T + j)*kd + kh*D])
          for d in 0 ..< D:
            dqrow[d] += ds * kr[d]
            dkr[d] += ds * qrow[d]

{.pop.}

proc causalAttention*(q, k, v: Tensor; B, T, nHeads, nKvHeads, headDim: int): Tensor =
  # Attention causale multi-têtes (avec têtes KV groupées).
  # q : [B*T, H*D], k et v : [B*T, Hkv*D]  ->  [B*T, H*D]
  result = newTensor([B*T, nHeads*headDim])
  var probs = newSeq[float32](B * nHeads * T * T)
  var c = AttCtx(q: fp(q.data), k: fp(k.data), v: fp(v.data), o: fp(result.data),
                 p: fp(probs), B: B, T: T, H: nHeads, Hkv: nKvHeads, D: headDim,
                 scale: 1'f32 / sqrt(headDim.float32))
  parallelFor(B * nHeads * T, attFwdTask, addr c)
  if needGrad(q, k, v):
    let r = result
    result.node(@[q, k, v], proc () =
      q.ensureGrad(); k.ensureGrad(); v.ensureGrad()
      var c2 = AttCtx(q: fp(q.data), k: fp(k.data), v: fp(v.data), p: fp(probs),
                      dq: fp(q.grad), dk: fp(k.grad), dv: fp(v.grad), gOut: fp(r.grad),
                      B: B, T: T, H: nHeads, Hkv: nKvHeads, D: headDim,
                      scale: 1'f32 / sqrt(headDim.float32))
      parallelFor(B * nKvHeads, attBwdTask, addr c2))

{.push checks: off, boundChecks: off.}
type CeCtx = object
  logits, grad: F
  targets: ptr UncheckedArray[int]
  V: int
  losses: F

proc ceTask(pp: pointer; first, last, worker: int) {.nimcall, gcsafe.} =
  let c = cast[ptr CeCtx](pp)
  let V = c.V
  for n in first ..< last:
    let t = c.targets[n]
    if t < 0:
      c.losses[n] = 0
      continue
    let row = cast[F](addr c.logits[n*V])
    var mx = -Inf.float32
    for i in 0 ..< V: mx = max(mx, row[i])
    var s = 0.0
    for i in 0 ..< V: s += exp(float(row[i] - mx))
    c.losses[n] = float32(ln(s) - float(row[t] - mx))

{.pop.}

proc crossEntropy*(logits: Tensor; targets: seq[int]): Tensor =
  # Entropie croisée moyenne. `targets[i] = -1` : position ignorée.
  # logits : [N, V]
  let V = logits.cols
  let N = logits.rows
  assert targets.len == N
  var losses = newSeq[float32](N)
  var tg = targets
  var c = CeCtx(logits: fp(logits.data), targets: cast[ptr UncheckedArray[int]](addr tg[0]),
                V: V, losses: fp(losses))
  parallelFor(N, ceTask, addr c)
  var total = 0.0
  var count = 0
  for n in 0 ..< N:
    if targets[n] >= 0:
      total += losses[n]; inc count
  result = scalar(float32(total / max(1, count).float))
  if needGrad(logits):
    let r = result
    result.node(@[logits], proc () =
      logits.ensureGrad()
      let g = r.grad[0] / max(1, count).float32
      for n in 0 ..< N:
        let t = targets[n]
        if t < 0: continue
        var mx = -Inf.float32
        for i in 0 ..< V: mx = max(mx, logits.data[n*V + i])
        var s = 0'f32
        for i in 0 ..< V: s += exp(logits.data[n*V + i] - mx)
        for i in 0 ..< V:
          let p = exp(logits.data[n*V + i] - mx) / s
          logits.grad[n*V + i] += g * (p - (if i == t: 1'f32 else: 0)))

proc concatCols*(a, b: Tensor): Tensor =
  # Concaténation horizontale [N, A] | [N, B] -> [N, A+B].
  let N = a.rows
  let A = a.cols
  let Bc = b.cols
  result = newTensor([N, A + Bc])
  for n in 0 ..< N:
    for i in 0 ..< A: result.data[n*(A+Bc) + i] = a.data[n*A + i]
    for i in 0 ..< Bc: result.data[n*(A+Bc) + A + i] = b.data[n*Bc + i]
  if needGrad(a, b):
    let r = result
    result.node(@[a, b], proc () =
      if a.requiresGrad:
        a.ensureGrad()
        for n in 0 ..< N:
          for i in 0 ..< A: a.grad[n*A + i] += r.grad[n*(A+Bc) + i]
      if b.requiresGrad:
        b.ensureGrad()
        for n in 0 ..< N:
          for i in 0 ..< Bc: b.grad[n*Bc + i] += r.grad[n*(A+Bc) + A + i])

proc dropout*(a: Tensor; p: float; rng: var Rand): Tensor =
  # Désactive aléatoirement une fraction `p` des valeurs (entraînement seulement).
  if p <= 0 or not gradEnabled: return a
  var mask = newSeq[float32](a.data.len)
  let k = float32(1.0 / (1.0 - p))
  for m in mask.mitems: m = (if rng.rand(1.0) < p: 0'f32 else: k)
  a * fromSeq(mask, a.shape)

proc gradCheck*(f: proc (): Tensor; t: Tensor; eps = 1e-2; samples = 10): float =
  # Compare le gradient analytique à une différence finie ; retourne l'écart
  # relatif maximal (outil de vérification pour vos propres opérations).
  t.zeroGrad()
  let loss = f()
  backward(loss)
  let g = t.grad
  var rng = initRand(7)
  for _ in 0 ..< samples:
    let i = rng.rand(t.data.high)
    let old = t.data[i]
    t.data[i] = old + eps
    var lp: float
    noGrad: lp = f().item.float
    t.data[i] = old - eps
    var lm: float
    noGrad: lm = f().item.float
    t.data[i] = old
    let num = (lp - lm) / (2*eps)
    let rel = abs(num - g[i].float) / max(1e-2, abs(num) + abs(g[i].float))
    result = max(result, rel)
