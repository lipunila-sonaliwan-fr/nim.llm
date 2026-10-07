# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/sampler — choix du token suivant à partir des logits.
#
# Chaîne appliquée (dans l'ordre) : pénalités de répétition -> biais ->
# température -> top-k -> top-p -> min-p -> tirage aléatoire.
# Une température <= 0 donne un décodage glouton (déterministe).

import std/[random, math, algorithm, tables, heapqueue]

type
  SamplerParams* = object
    temperature*: float          # 0 = glouton ; 0.7 typique ; > 1 plus créatif
    topK*: int                   # 0 = désactivé
    topP*: float                 # 1.0 = désactivé
    minP*: float                 # 0 = désactivé
    repeatPenalty*: float        # 1.0 = désactivé ; 1.1 typique
    repeatLastN*: int            # fenêtre des tokens pénalisés
    presencePenalty*: float
    frequencyPenalty*: float
    seed*: int                   # -1 = aléatoire
    logitBias*: Table[int, float]# biais additif par token (-Inf = interdit)

  Sampler* = ref object
    params*: SamplerParams
    rng*: Rand
    history*: seq[int]

proc defaultSampling*(): SamplerParams =
  # Paramètres raisonnables pour la conversation.
  SamplerParams(temperature: 0.7, topK: 40, topP: 0.95, minP: 0.05,
                repeatPenalty: 1.1, repeatLastN: 64, seed: -1)

proc greedySampling*(): SamplerParams =
  # Décodage déterministe (toujours le token le plus probable).
  SamplerParams(temperature: 0, topK: 0, topP: 1, repeatPenalty: 1.0, seed: 0)

proc newSampler*(p = defaultSampling()): Sampler =
  result = Sampler(params: p)
  result.rng = if p.seed < 0: initRand() else: initRand(p.seed)

proc accept*(s: Sampler; tok: int) =
  # Mémorise un token (pour les pénalités de répétition).
  s.history.add tok

proc argmax*(logits: openArray[float32]): int =
  var best = 0
  for i in 1 ..< logits.len:
    if logits[i] > logits[best]: best = i
  best

proc softmaxInPlace*(x: var seq[float32]) =
  var mx = -Inf.float32
  for v in x: mx = max(mx, v)
  var s = 0'f32
  for v in x.mitems:
    v = exp(v - mx); s += v
  for v in x.mitems: v /= s

proc sample*(s: Sampler; logitsIn: openArray[float32]): int =
  # Choisit un token. Ne modifie pas `logitsIn`.
  let p = s.params
  var logits = @logitsIn
  # pénalités.
  if s.history.len > 0 and (p.repeatPenalty != 1.0 or p.presencePenalty != 0 or
                             p.frequencyPenalty != 0):
    var counts = initCountTable[int]()
    let start = if p.repeatLastN > 0: max(0, s.history.len - p.repeatLastN) else: 0
    for i in start ..< s.history.len: counts.inc s.history[i]
    for tok, cnt in counts:
      if tok < 0 or tok >= logits.len: continue
      if p.repeatPenalty != 1.0:
        if logits[tok] > 0: logits[tok] /= p.repeatPenalty.float32
        else: logits[tok] *= p.repeatPenalty.float32
      logits[tok] -= float32(cnt.float * p.frequencyPenalty + p.presencePenalty)
  for tok, b in p.logitBias:
    if tok >= 0 and tok < logits.len: logits[tok] += b.float32
  if p.temperature <= 0:
    return argmax(logits)
  # température.
  let invT = float32(1.0 / p.temperature)
  for v in logits.mitems: v *= invT
  # candidats triés.
  var idx = newSeq[int](logits.len)
  for i in 0 ..< idx.len: idx[i] = i
  let k = if p.topK > 0: min(p.topK, logits.len) else: logits.len
  if k < logits.len:
    # sélection partielle (tas de taille k) : évite de trier tout le vocabulaire.
    var h = initHeapQueue[(float32, int)]()
    for i in 0 ..< logits.len:
      if h.len < k: h.push((logits[i], i))
      elif logits[i] > h[0][0]:
        discard h.replace((logits[i], i))
    idx.setLen(0)
    while h.len > 0: idx.add h.pop()[1]
    idx.reverse()
  else:
    idx.sort(proc(a, b: int): int = cmp(logits[b], logits[a]))
  var probs = newSeq[float32](idx.len)
  for i, id in idx: probs[i] = logits[id]
  softmaxInPlace(probs)
  var n = probs.len
  # top-p
  if p.topP > 0 and p.topP < 1:
    var cum = 0'f32
    for i in 0 ..< n:
      cum += probs[i]
      if cum >= p.topP:
        n = i + 1; break
  # min-p
  if p.minP > 0:
    let thr = probs[0] * p.minP.float32
    var m = 1
    while m < n and probs[m] >= thr: inc m
    n = m
  var total = 0'f32
  for i in 0 ..< n: total += probs[i]
  var r = s.rng.rand(1.0).float32 * total
  for i in 0 ..< n:
    r -= probs[i]
    if r <= 0: return idx[i]
  idx[n - 1]

proc topTokens*(logits: openArray[float32]; n = 5): seq[(int, float32)] =
  # Les `n` tokens les plus probables avec leur probabilité.
  var probs = @logits
  softmaxInPlace(probs)
  var idx = newSeq[int](probs.len)
  for i in 0 ..< idx.len: idx[i] = i
  idx.sort(proc(a, b: int): int = cmp(probs[b], probs[a]))
  for i in 0 ..< min(n, idx.len): result.add (idx[i], probs[idx[i]])
