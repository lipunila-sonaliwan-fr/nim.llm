# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/train — optimiseurs, planification du taux d'apprentissage,
# jeux de données et boucle d'entraînement.

import std/[math, random, strutils, json, times]
import autograd, nn, tokenizer, chat, gguf

#
# Optimiseurs
#

type
  OptimizerKind* = enum okAdamW = "adamw", okSGD = "sgd"

  Optimizer* = ref object
    # AdamW (par défaut) ou SGD avec moment.
    kind*: OptimizerKind
    params*: seq[Tensor]
    lr*: float
    beta1*, beta2*, eps*: float
    weightDecay*: float
    momentum*: float
    step*: int
    m, v: seq[seq[float32]]

proc newAdamW*(params: seq[Tensor]; lr = 3e-4; beta1 = 0.9; beta2 = 0.95;
               eps = 1e-8; weightDecay = 0.1): Optimizer =
  # AdamW : Adam avec décroissance des poids découplée (appliquée aux matrices).
  result = Optimizer(kind: okAdamW, params: params, lr: lr, beta1: beta1, beta2: beta2,
                     eps: eps, weightDecay: weightDecay)
  for p in params:
    result.m.add newSeq[float32](p.numel)
    result.v.add newSeq[float32](p.numel)

proc newSGD*(params: seq[Tensor]; lr = 0.01; momentum = 0.9; weightDecay = 0.0): Optimizer =
  result = Optimizer(kind: okSGD, params: params, lr: lr, momentum: momentum,
                     weightDecay: weightDecay)
  for p in params: result.m.add newSeq[float32](p.numel)

proc zeroGrad*(o: Optimizer) =
  # Remet à zéro les gradients des paramètres.
  for p in o.params: p.zeroGrad()

proc gradNorm*(params: seq[Tensor]): float =
  for p in params:
    for g in p.grad: result += float(g) * float(g)
  sqrt(result)

proc clipGradNorm*(params: seq[Tensor]; maxNorm: float): float =
  # Limite la norme globale des gradients ; retourne la norme avant écrêtage.
  result = gradNorm(params)
  if maxNorm > 0 and result > maxNorm:
    let k = float32(maxNorm / (result + 1e-6))
    for p in params:
      for g in p.grad.mitems: g *= k

proc update*(o: Optimizer) =
  # Applique une étape d'optimisation à partir des gradients courants.
  inc o.step
  case o.kind
  of okAdamW:
    let b1 = o.beta1.float32
    let b2 = o.beta2.float32
    let bc1 = float32(1 - pow(o.beta1, o.step.float))
    let bc2 = float32(1 - pow(o.beta2, o.step.float))
    let lr = o.lr.float32
    let eps = o.eps.float32
    for i, p in o.params:
      if p.grad.len == 0: continue
      let wd = if p.shape.len >= 2: float32(o.weightDecay) else: 0'f32
      for j in 0 ..< p.data.len:
        let g = p.grad[j]
        o.m[i][j] = b1 * o.m[i][j] + (1 - b1) * g
        o.v[i][j] = b2 * o.v[i][j] + (1 - b2) * g * g
        let mh = o.m[i][j] / bc1
        let vh = o.v[i][j] / bc2
        p.data[j] -= lr * (mh / (sqrt(vh) + eps) + wd * p.data[j])
  of okSGD:
    let lr = o.lr.float32
    let mom = o.momentum.float32
    for i, p in o.params:
      if p.grad.len == 0: continue
      for j in 0 ..< p.data.len:
        let g = p.grad[j] + float32(o.weightDecay) * p.data[j]
        o.m[i][j] = mom * o.m[i][j] + g
        p.data[j] -= lr * o.m[i][j]

proc saveState*(o: Optimizer; path: string) =
  # Enregistre l'état de l'optimiseur (moments) pour reprendre l'entraînement.
  let w = newGgufWriter()
  w.setKV("optimizer.kind", gStr($o.kind))
  w.setKV("optimizer.step", gU64(o.step))
  for i, p in o.params:
    w.addTensorF32("m." & $i, @[o.m[i].len], o.m[i])
    if o.kind == okAdamW: w.addTensorF32("v." & $i, @[o.v[i].len], o.v[i])
  w.write(path)

proc loadState*(o: Optimizer; path: string) =
  let g = openGguf(path)
  o.step = g.getInt("optimizer.step")
  for i in 0 ..< o.params.len:
    if "m." & $i in g.tensors: o.m[i] = g.tensorF32("m." & $i)
    if o.kind == okAdamW and "v." & $i in g.tensors: o.v[i] = g.tensorF32("v." & $i)
  g.close()

proc cosineLr*(step, warmup, total: int; maxLr: float; minLr = 0.0): float =
  # Montée linéaire pendant `warmup` étapes puis décroissance en cosinus.
  if step < warmup: return maxLr * (step + 1).float / warmup.float
  if step >= total: return minLr
  let p = (step - warmup).float / max(1, total - warmup).float
  minLr + 0.5 * (maxLr - minLr) * (1 + cos(PI * p))

#
# Jeux de données
#

type
  Batch* = object
    inputs*, targets*: seq[int]  # B*T identifiants ; cible -1 = ignorée.
    B*, T*: int

  DatasetKind* = enum dkText, dkChat

  Example* = object
    tokens*: seq[int]
    mask*: seq[bool]             # true = token à apprendre.

  Dataset* = ref object
    # Données d'entraînement : flux de texte continu ou exemples de dialogue.
    kind*: DatasetKind
    tokens*: seq[int]            # dkText
    examples*: seq[Example]      # dkChat
    padId*: int
    rng*: Rand

proc newTextDataset*(tok: Tokenizer; text: string; seed = 1; addBos = true): Dataset =
  # Jeu de données « modèle de langage » à partir d'un texte brut.
  # Les documents peuvent être séparés par des lignes vides doubles.
  result = Dataset(kind: dkText, rng: initRand(seed))
  for doc in text.split("\n\n\n"):
    if doc.strip.len == 0: continue
    if addBos and tok.bosId >= 0: result.tokens.add tok.bosId
    result.tokens.add tok.encode(doc, parseSpecial = false)
    if tok.eosId >= 0: result.tokens.add tok.eosId

proc newTextDatasetFromFiles*(tok: Tokenizer; paths: seq[string]; seed = 1): Dataset =
  var all = ""
  for p in paths: all.add readFile(p) & "\n\n\n"
  newTextDataset(tok, all, seed)

proc len*(d: Dataset): int =
  if d.kind == dkText: d.tokens.len else: d.examples.len

proc split*(d: Dataset; valFraction = 0.1): (Dataset, Dataset) =
  # Sépare en (entraînement, validation).
  var a = Dataset(kind: d.kind, padId: d.padId, rng: initRand(d.rng.rand(1_000_000)))
  var b = Dataset(kind: d.kind, padId: d.padId, rng: initRand(d.rng.rand(1_000_000)))
  if d.kind == dkText:
    let cut = int(d.tokens.len.float * (1 - valFraction))
    a.tokens = d.tokens[0 ..< cut]
    b.tokens = d.tokens[cut .. ^1]
  else:
    var ex = d.examples
    var r = initRand(17)
    r.shuffle(ex)
    let cut = max(1, int(ex.len.float * (1 - valFraction)))
    a.examples = ex[0 ..< cut]
    b.examples = if cut < ex.len: ex[cut .. ^1] else: ex[0 ..< 1]
  (a, b)

proc addChatExample*(d: Dataset; tok: Tokenizer; templ: ChatTemplateKind;
                     msgs: seq[Message]) =
  # Ajoute un dialogue ; seules les réponses de l'assistant sont apprises.
  var ex: Example
  for i, m in msgs:
    # préfixe jusqu'au message i (inclus) : on tokenise incrémentalement.
    let before = encodeMessages(tok, templ, msgs[0 ..< i], addGenerationPrompt = m.role == roleAssistant)
    let upto = encodeMessages(tok, templ, msgs[0 .. i], addGenerationPrompt = false)
    if ex.tokens.len == 0:
      ex.tokens = before
      ex.mask = newSeq[bool](before.len)
    # complète jusqu'à `before` (balise d'ouverture de la réponse).
    while ex.tokens.len < before.len:
      ex.tokens.add before[ex.tokens.len]; ex.mask.add false
    for k in ex.tokens.len ..< upto.len:
      ex.tokens.add upto[k]
      ex.mask.add(m.role == roleAssistant)
  d.examples.add ex

proc newChatDataset*(tok: Tokenizer; templ: ChatTemplateKind;
                     conversations: seq[seq[Message]]; seed = 1): Dataset =
  # Jeu de données de dialogues (ajustement « instruction »/SFT).
  result = Dataset(kind: dkChat, rng: initRand(seed),
                   padId: (if tok.eosId >= 0: tok.eosId else: 0))
  for conv in conversations: result.addChatExample(tok, templ, conv)

proc loadChatJsonl*(path: string; tok: Tokenizer; templ: ChatTemplateKind;
                    system = ""; seed = 1): Dataset =
  # Charge un fichier JSON : une conversation par ligne, au format
  # `{"messages": [{"role": "user", "content": "..."}, ...]}`
  # ou `{"prompt": "...", "response": "..."}` / `{"instruction": ..., "output": ...}`.
  var convs: seq[seq[Message]]
  for line in lines(path):
    if line.strip.len == 0: continue
    let j = parseJson(line)
    var conv: seq[Message]
    if system.len > 0: conv.add Message(role: roleSystem, content: system)
    if j.hasKey("messages"):
      for m in j["messages"]:
        let r = case m["role"].getStr
          of "system": roleSystem
          of "assistant": roleAssistant
          else: roleUser
        conv.add Message(role: r, content: m["content"].getStr)
    else:
      let q = j{"prompt"}.getStr(j{"instruction"}.getStr(j{"question"}.getStr))
      let inp = j{"input"}.getStr
      let a = j{"response"}.getStr(j{"output"}.getStr(j{"answer"}.getStr))
      conv.add Message(role: roleUser, content: (if inp.len > 0: q & "\n\n" & inp else: q))
      conv.add Message(role: roleAssistant, content: a)
    convs.add conv
  newChatDataset(tok, templ, convs, seed)

proc getBatch*(d: Dataset; B, T: int): Batch =
  # Tire un lot aléatoire de B séquences de longueur T.
  result = Batch(B: B, T: T)
  result.inputs = newSeq[int](B*T)
  result.targets = newSeq[int](B*T)
  case d.kind
  of dkText:
    if d.tokens.len < T + 2:
      raise newException(ValueError, "texte trop court (" & $d.tokens.len &
        " tokens) pour des séquences de " & $T)
    for b in 0 ..< B:
      let s = d.rng.rand(d.tokens.len - T - 2)
      for t in 0 ..< T:
        result.inputs[b*T + t] = d.tokens[s + t]
        result.targets[b*T + t] = d.tokens[s + t + 1]
  of dkChat:
    if d.examples.len == 0: raise newException(ValueError, "aucun exemple")
    for b in 0 ..< B:
      let ex = d.examples[d.rng.rand(d.examples.high)]
      for t in 0 ..< T:
        if t < ex.tokens.len:
          result.inputs[b*T + t] = ex.tokens[t]
          result.targets[b*T + t] =
            if t + 1 < ex.tokens.len and ex.mask[t + 1]: ex.tokens[t + 1] else: -1
        else:
          result.inputs[b*T + t] = d.padId
          result.targets[b*T + t] = -1

#
# Boucle d'entraînement
#

type
  TrainConfig* = object
    steps*: int                  # nombre d'étapes d'optimisation.
    batchSize*: int              # B
    seqLen*: int                 # T
    gradAccum*: int              # lots accumulés par étape.
    lr*: float                    # taux d'apprentissage maximal.
    minLr*: float
    warmup*: int
    weightDecay*: float
    gradClip*: float
    evalEvery*: int              # 0 = jamais.
    evalBatches*: int
    logEvery*: int
    saveEvery*: int              # 0 = jamais (checkpoint).
    savePath*: string            # préfixe des fichiers de sauvegarde.
    sampleEvery*: int            # affiche un exemple de génération.
    samplePrompt*: string
    seed*: int

  TrainLog* = object
    step*: int
    loss*: float
    valLoss*: float              # NaN si pas d'évaluation à cette étape.
    lr*: float
    gradNorm*: float
    tokensPerSec*: float
    elapsed*: float

  LogCallback* = proc (log: TrainLog) {.closure.}

proc defaultTrainConfig*(): TrainConfig =
  TrainConfig(steps: 200, batchSize: 8, seqLen: 64, gradAccum: 1, lr: 1e-3, minLr: 1e-4,
              warmup: 20, weightDecay: 0.1, gradClip: 1.0, evalEvery: 50, evalBatches: 4,
              logEvery: 10, saveEvery: 0, savePath: "checkpoint", sampleEvery: 0,
              samplePrompt: "", seed: 1)

proc evaluate*(t: Transformer; d: Dataset; batches = 4; B = 8; T = 64): float =
  # Perte moyenne (entropie croisée) sur `batches` lots, sans gradient.
  var total = 0.0
  noGrad:
    for i in 0 ..< batches:
      let b = d.getBatch(B, T)
      total += t.loss(b.inputs, b.targets, B, T).item
  total / batches.float

proc saveCheckpoint*(t: Transformer; opt: Optimizer; prefix: string) =
  # Sauvegarde le modèle (ou l'adaptateur LoRA) et l'état de l'optimiseur.
  if t.mode == tmLora: t.saveLora(prefix & ".lora.gguf")
  else: t.saveGguf(prefix & ".gguf")
  if opt != nil: opt.saveState(prefix & ".optim.gguf")

proc printLog*(l: TrainLog) =
  # Affichage standard d'une ligne de journal.
  var s = "étape " & align($l.step, 5) & " | perte " & formatFloat(l.loss, ffDecimal, 4)
  if not l.valLoss.isNaN: s.add " | val " & formatFloat(l.valLoss, ffDecimal, 4)
  s.add " | lr " & formatFloat(l.lr, ffScientific, 2) &
        " | ‖g‖ " & formatFloat(l.gradNorm, ffDecimal, 2) &
        " | " & $int(l.tokensPerSec) & " tok/s"
  echo s

proc train*(t: Transformer; data: Dataset; cfg = defaultTrainConfig();
            valData: Dataset = nil; opt: Optimizer = nil;
            onLog: LogCallback = nil): Optimizer =
  # Entraîne le modèle. Retourne l'optimiseur (pour poursuivre plus tard).
  let params = t.parameters
  if params.len == 0: raise newException(ValueError, "aucun paramètre entraînable")
  let o = if opt != nil: opt else: newAdamW(params, cfg.lr, weightDecay = cfg.weightDecay)
  if data.kind == dkChat:
    var trop, plusLong = 0
    for ex in data.examples:
      plusLong = max(plusLong, ex.tokens.len)
      if ex.tokens.len > cfg.seqLen: inc trop
    if trop > 0:
      stderr.writeLine "[nimllm] attention : ", trop, " exemple(s) sur ", data.examples.len,
        " dépassent seqLen = ", cfg.seqLen, " tokens (le plus long : ", plusLong,
        ") ; leur fin sera ignorée. Augmentez seqLen."
  if cfg.seqLen > t.cfg.ctxTrain:
    raise newException(ValueError, "seqLen (" & $cfg.seqLen & ") dépasse le contexte du modèle (" &
      $t.cfg.ctxTrain & ")")
  var log: LogCallback = onLog
  if log == nil:
    log = proc (l: TrainLog) = printLog(l)
  let t0 = epochTime()
  var tokCount = 0
  var tLast = epochTime()
  let accum = max(1, cfg.gradAccum)
  let startStep = o.step
  for step in startStep ..< startStep + cfg.steps:
    o.lr = cosineLr(step - startStep, cfg.warmup, cfg.steps, cfg.lr, cfg.minLr)
    o.zeroGrad()
    var lossSum = 0.0
    for a in 0 ..< accum:
      let b = data.getBatch(cfg.batchSize, cfg.seqLen)
      var loss = t.loss(b.inputs, b.targets, b.B, b.T)
      lossSum += loss.item
      if accum > 1: loss = loss.scale(1'f32 / accum.float32)
      backward(loss)
      tokCount += b.B * b.T
    let gn = clipGradNorm(params, cfg.gradClip)
    o.update()
    let s = step - startStep + 1
    var val = NaN
    if valData != nil and cfg.evalEvery > 0 and (s mod cfg.evalEvery == 0 or s == cfg.steps):
      val = t.evaluate(valData, cfg.evalBatches, cfg.batchSize, cfg.seqLen)
    if cfg.logEvery > 0 and (s mod cfg.logEvery == 0 or s == 1 or s == cfg.steps or not val.isNaN):
      let now = epochTime()
      log TrainLog(step: o.step, loss: lossSum / accum.float, valLoss: val, lr: o.lr,
                   gradNorm: gn, tokensPerSec: tokCount.float / max(1e-9, now - tLast),
                   elapsed: now - t0)
      tokCount = 0
      tLast = now
    if cfg.sampleEvery > 0 and s mod cfg.sampleEvery == 0:
      echo "   » ", t.generateText(cfg.samplePrompt, 40).replace("\n", "⏎")
    if cfg.saveEvery > 0 and s mod cfg.saveEvery == 0:
      t.saveCheckpoint(o, cfg.savePath)
  o
