# 14 — Référence de l'API

`import nimllm` donne accès à tous les modules ci-dessous. Les signatures sont
données en notation Nim ; les paramètres avec `=` sont optionnels.

## chat — conversation

### Types

```nim
type
  Role = enum roleSystem, roleUser, roleAssistant
  Message = object
    role: Role
    content: string
  ChatTemplateKind = enum tplAuto, tplLlama3, tplChatML, tplMistral, tplLlama2, tplGemma, tplPhi3, tplRaw
  OutputFormat = enum ofText, ofMarkdown, ofJson, ofImage, ofAudio, ofFile
  StopReason = enum srEndOfText, srStopString, srMaxTokens, srContextFull, srCallback
  TokenCallback = proc (piece: string): bool {.closure.}
  GenOptions = object
    maxTokens: int; sampling: SamplerParams; stop: seq[string]; onToken: TokenCallback
  Reply = object
    text, raw: string; files: seq[string]; json: JsonNode
    promptTokens, completionTokens: int; stopReason: StopReason
    seconds, tokensPerSecond: float
  Chat = ref object
    model: LlmModel; ctx: LlmContext; system: string; history: seq[Message]
    templ: ChatTemplateKind; options: GenOptions; stripThinking: bool
    lang: string; imageSize: int; jsonRetries: int; voicePitch: float
```

### Fonctions

| Signature | Description |
|---|---|
| `newChat(m; system = ""; nCtx = 4096; templ = tplAuto; sampling = defaultSampling(); maxTokens = 512; lang = "fr"): Chat` | nouvelle conversation |
| `ask(c; prompt; attachments = @[]; format = ofText; outPath = ""; schema = ""; onToken = nil; maxTokens = 0): Reply` | pose une question |
| `askJson(c; prompt; schema = ""): JsonNode` | question à réponse JSON |
| `regenerate(c; onToken = nil): Reply` | nouveau tirage de la dernière réponse |
| `continueReply(c; maxTokens = 256; onToken = nil): Reply` | prolonge la dernière réponse |
| `add(c; role; content)` | ajoute un message sans générer |
| `reset(c)` / `undo(c)` | efface l'historique / retire le dernier échange |
| `setSampling(c; p)` | change les réglages d'échantillonnage |
| `messages(c): seq[Message]` | système + historique |
| `promptTokens(c; addGen = true): seq[int]` | prompt tokenisé |
| `saveHistory(c; path)` / `loadHistory(c; path)` / `toJson(c)` | persistance |
| `stats(c): string` | état du contexte |
| `generate(ctx; prompt: seq[int]; opts; smp = nil): Reply` | génération à partir de tokens |
| `complete(ctx; text; opts = defaultOptions()): Reply` | complétion brute |
| `defaultOptions(): GenOptions` | 512 tokens, `defaultSampling()` |
| `detectTemplate(tok): ChatTemplateKind` | gabarit d'un modèle |
| `renderPrompt(kind; msgs; addGenerationPrompt = true; bos = "<s>"): string` | texte du prompt |
| `encodeMessages(tok; kind; msgs; addGenerationPrompt = true): seq[int]` | prompt tokenisé (balises sûres) |
| `stopStringsFor(kind): seq[string]` | chaînes d'arrêt implicites |
| `extractJson(s): JsonNode` | extrait le premier JSON d'un texte |
| `stripFences(s): string` | retire les balises ``` |

## model — inférence

| Signature | Description |
|---|---|
| `loadModel(path; threads = 0; verbose = false): LlmModel` | ouvre un GGUF |
| `describe(m): string` / `paramCount(m): int` / `close(m)` | informations / fermeture |
| `m.cfg: ModelConfig` | `arch, name, dim, hidden, nLayers, nHeads, nKvHeads, headDim, vocab, ctxTrain, ropeBase, ropeDim, ropeNeox, normEps, tiedEmbeddings` |
| `m.tokenizer: Tokenizer` | tokeniseur du modèle |
| `newContext(m; nCtx = 2048; nBatch = 32): LlmContext` | cache KV |
| `eval(ctx; toks; allLogits = false): seq[float32]` | évalue des tokens |
| `evalPrompt(ctx; toks): seq[float32]` | idem avec réutilisation du préfixe en cache |
| `reset(ctx)` / `truncate(ctx; n)` / `nPast(ctx)` / `ctx.tokens` | gestion du cache |
| `memoryUsage(ctx)` / `tokensPerSecond(ctx)` | statistiques |
| `applyLora(m; path)` / `removeLora(m)` | adaptateurs à la volée |
| `embed(m; text; nCtx = 512): seq[float32]` | vecteur normalisé |
| `cosineSimilarity(a, b): float` | similarité |
| `perplexity(m; text; nCtx = 512): float` | perplexité |
| `exactMatmul: bool` (variable globale) | désactive les activations int8 |
| `newQMatrix(name; typ; rows, cols; values): QMatrix` / `matmul(w; x; y; nb)` | matrices quantifiées |

## sampler — échantillonnage

```nim
type SamplerParams = object
  temperature, topP, minP, repeatPenalty, presencePenalty, frequencyPenalty: float
  topK, repeatLastN, seed: int
  logitBias: Table[int, float]
```

| Signature | Description |
|---|---|
| `defaultSampling()` / `greedySampling()` | préréglages |
| `newSampler(p): Sampler` | échantillonneur |
| `sample(s; logits): int` / `accept(s; tok)` | tirage / mémorisation |
| `argmax(logits)` / `softmaxInPlace(x)` / `topTokens(logits; n = 5)` | utilitaires |

## tokenizer — tokens

| Signature | Description |
|---|---|
| `encode(t; text; addBos = false; parseSpecial = true): seq[int]` | texte → tokens |
| `decode(t; ids; renderSpecial = false): string` | tokens → texte |
| `tokenToPiece(t; id; renderSpecial = false): string` | un token |
| `tokenId(t; text): int` / `isSpecial(t; id)` / `vocabSize(t)` | recherche |
| `t.bosId, t.eosId, t.padId, t.unkId, t.eogIds, t.addBos, t.chatTemplate, t.tokens, t.merges, t.kind, t.pre` | champs |
| `tokenizerFromGguf(g)` / `writeToGguf(t; w)` | lecture / écriture GGUF |
| `newBpeTokenizer(tokens; merges; types; pre = ptLlama3)` / `rebuild(t)` | construction |
| `preTokenize(text; mode)` / `byteEncode(s)` / `byteDecode(s)` | outils BPE |

## tokentrain — apprentissage de tokeniseur

| Signature | Description |
|---|---|
| `trainBpe(texts; vocabSize = 2048; specials = defaultSpecials; minFreq = 2; pre = ptLlama3; verbose = false): Tokenizer` | BPE niveau octet |
| `byteLevelTokenizer(specials = defaultSpecials): Tokenizer` | un token par octet |
| `defaultSpecials` / `chatmlTemplate` | constantes |

## attachments — pièces jointes

```nim
type Attachment = object
  name, mime, text, data: string
  kind: AttachmentKind      # akText, akImage, akPdf, akAudio, akBinary
  width, height: int
  image: Image              # pixels si PNG/BMP/PPM
```

| Signature | Description |
|---|---|
| `attach(path): Attachment` | depuis un fichier |
| `attachText(name; content)` / `attachData(name; data)` | depuis la mémoire |
| `toPrompt(a; maxChars = 12000): string` | texte inséré dans le message |
| `decodePng(data): Image` / `pdfText(data): string` / `describeImage(img)` | décodeurs |

## image — images

| Signature | Description |
|---|---|
| `rgb(r, g, b; a = 1.0): Color` / `parseColor(s)` | couleurs |
| `newImage(w, h; bg)` / `getPixel` / `setPixel` | image RVB |
| `writeBmp` / `writePpm` / `save` / `readBmp` / `readPpm` | fichiers |
| `newCanvas(w, h; bg; ss = 3): Canvas` / `finish(c): Image` | dessin anticrénelé |
| `fillRect`, `fillCircle`, `fillPolygon`, `fillPolygons`, `strokePolyline`, `drawLine`, `ellipsePoints` | primitives |
| `renderSvg(svg; width = 0; height = 0; bg): Image` | rendu SVG |
| `extractSvg(text)` / `svgSize(svg)` / `parsePath(d)` | outils SVG |

## audio — sons

| Signature | Description |
|---|---|
| `speak(text; lang = "fr"; pitch = 115.0; speed = 1.0; sampleRate = 16000): Audio` | synthèse vocale |
| `renderMelody(notation; bpm = 120.0; sampleRate = 22050): Audio` | mélodie |
| `writeWav(a; path)` / `readWav(path)` / `readWavInfo(path)` | fichiers WAV |
| `concat(a; b)` / `silence(a; secondes)` / `normalize(a)` / `duration(a)` | montage |
| `textToPhonemes(text; lang)` / `numberToFrench(n)` / `numberToEnglish(n)` / `noteFrequency(note)` | outils |

## autograd — tenseurs différentiables

| Signature | Description |
|---|---|
| `newTensor(shape; requiresGrad = false)`, `fromSeq(data; shape)`, `full(shape; v)`, `randn(shape; std; rng)`, `param(shape; std; rng; name)`, `scalar(v)` | création |
| `t.data`, `t.grad`, `t.shape`, `rows`, `cols`, `numel`, `item`, `reshape`, `detach` | accès |
| `+`, `-`, `*`, `scale`, `sum`, `mean`, `linear(x, w, b = nil)`, `matmul`, `concatCols` | opérations |
| `relu`, `silu`, `gelu`, `sigmoid`, `tanhT`, `softmax`, `rmsnorm`, `dropout` | fonctions |
| `embedding`, `rope`, `causalAttention`, `crossEntropy`, `mseLoss` | blocs de LLM |
| `backward(loss)` / `zeroGrad(t)` / `noGrad: …` | gradients |
| `needsGrad(…)` / `makeNode(res; parents; backward)` / `ensureGrad(t)` | opérations personnalisées |
| `gradCheck(f; t; eps = 1e-2; samples = 10): float` | vérification |

## nn — modèles entraînables

| Signature | Description |
|---|---|
| `newModelConfig(vocab; dim = 256; layers = 4; heads = 4; kvHeads = 0; hidden = 0; ctx = 256; ropeBase = 10000.0; tied = true; name): ModelConfig` | architecture |
| `newTransformer(cfg; tok; seed = 42): Transformer` | modèle neuf |
| `loadForTraining(path; mode = tmLora; lora = defaultLora(); seed = 42; threads = 0): Transformer` | depuis un GGUF |
| `defaultLora(): LoraConfig` (`rank`, `alpha`, `targets`) | réglages LoRA |
| `parameters(t)` / `parameterCount(t)` | paramètres |
| `forward(t; ids; B, T)` / `hidden(t; ids; B, T)` / `loss(t; inputs; targets; B, T)` | calcul |
| `generateText(t; prompt; maxTokens = 50; temperature = 0.8)` | génération de contrôle |
| `saveGguf(t; path; wtype = gtF32)` / `saveLora(t; path)` | export |
| `mergeLora(base; adapter; out; outType = gtQ8_0)` / `quantizeModel(in; out; wtype; keepOutput = true)` | fichiers |
| `qlinear(x; q)` / `qembedding(q; ids)` / `Linear.forward(x)` / `addLora(l; rank; alpha; rng)` | couches |

## train — entraînement

| Signature | Description |
|---|---|
| `newAdamW(params; lr = 3e-4; beta1 = 0.9; beta2 = 0.95; eps = 1e-8; weightDecay = 0.1)` / `newSGD(params; lr; momentum; weightDecay)` | optimiseurs |
| `zeroGrad(o)` / `update(o)` / `saveState(o; path)` / `loadState(o; path)` / `o.lr`, `o.step` | optimiseur |
| `clipGradNorm(params; maxNorm)` / `gradNorm(params)` / `cosineLr(step, warmup, total, maxLr, minLr)` | outils |
| `newTextDataset(tok; text)` / `newTextDatasetFromFiles(tok; paths)` | données texte |
| `newChatDataset(tok; templ; conversations)` / `loadChatJsonl(path; tok; templ; system = "")` / `addChatExample` | dialogues |
| `split(d; valFraction = 0.1)` / `getBatch(d; B, T): Batch` / `len(d)` | manipulation |
| `defaultTrainConfig(): TrainConfig` | `steps, batchSize, seqLen, gradAccum, lr, minLr, warmup, weightDecay, gradClip, evalEvery, evalBatches, logEvery, saveEvery, savePath, sampleEvery, samplePrompt, seed` |
| `train(t; data; cfg; valData = nil; opt = nil; onLog = nil): Optimizer` | boucle complète |
| `evaluate(t; d; batches = 4; B = 8; T = 64): float` | perte moyenne |
| `saveCheckpoint(t; opt; prefix)` / `printLog(l)` | sauvegarde / journal |

## gguf et quant — fichiers et formats

| Signature | Description |
|---|---|
| `openGguf(path): GgufFile` / `close(g)` / `describe(g)` | lecture |
| `g.kv`, `g.tensors`, `has`, `getInt`, `getFloat`, `getStr`, `getBool`, `getStrArray`, `getIntArray`, `getFloatArray` | métadonnées |
| `tensorF32(g; name)`, `GgufTensorInfo.dims/typ/data/numElements/byteSize` | tenseurs |
| `newGgufWriter()`, `setKV`, `addTensor`, `addTensorF32(name; dims; values; typ)`, `addTensorRaw`, `copyMetadata`, `write` | écriture |
| `gStr`, `gU32`, `gI32`, `gU64`, `gF32`, `gBool`, `gArrStr`, `gArrI32`, `gArrF32` | valeurs |
| `GgmlType` (`gtF32`, `gtF16`, `gtBF16`, `gtQ8_0`, `gtQ4_0`… `gtQ6_K`), `parseGgmlType("q4_k")` | types |
| `quantize(t; data)`, `quantizeRow`, `dequantRow`, `blockSize`, `typeSize`, `rowBytes`, `bitsPerWeight` | conversions |
| `floatToHalf` / `halfToFloat` / `floatToBf16` / `bf16ToFloat` | demi-précision |

## parallel — threads

| Signature | Description |
|---|---|
| `setThreads(n = 0)` / `numThreads()` / `shutdownPool()` | pool |
| `parallelFor(n; fn: TaskFn; ctx: pointer; minChunk = 1)` | boucle parallèle (`fn(ctx, first, last, worker)`) |

## inflate — décompression

| Signature | Description |
|---|---|
| `zlibDecompress(data)` / `inflateRaw(data)` | flux zlib / DEFLATE |

<br/><br/>

> © 2026 Jean-Marc Quéré, sonaliwan.fr - Linguistique & Technologies<br/>
Tous droits réservés<br/>
SIRET : 123 456 789 00012<br/>
Licence : CC BY-NC-SA 4.0<br/>
