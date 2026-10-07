# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/chat — dialogue avec un modèle : contexte (prompt système),
# historique, pièces jointes, génération en flux et formats de réponse
# (texte, Markdown, JSON, image SVG/BMP, audio WAV, fichier).

import std/[strutils, json, os, times, unicode, sets]
import model, tokenizer, sampler, attachments, image, audio

type
  Role* = enum
    roleSystem = "system", roleUser = "user", roleAssistant = "assistant"

  Message* = object
    role*: Role
    content*: string

  ChatTemplateKind* = enum
    tplAuto = "auto", tplLlama3 = "llama3", tplChatML = "chatml",
    tplMistral = "mistral", tplLlama2 = "llama2", tplGemma = "gemma",
    tplPhi3 = "phi3", tplRaw = "raw"

  OutputFormat* = enum
    ofText = "texte", ofMarkdown = "markdown", ofJson = "json",
    ofImage = "image", ofAudio = "audio", ofFile = "fichier"

  StopReason* = enum
    srEndOfText = "fin de texte", srStopString = "chaîne d'arrêt",
    srMaxTokens = "limite de tokens", srContextFull = "contexte plein",
    srCallback = "interrompu"

  TokenCallback* = proc (piece: string): bool {.closure.}
    # Reçoit chaque morceau de texte généré ; retourner `false` interrompt.

  GenOptions* = object
    maxTokens*: int              # nombre maximal de tokens générés.
    sampling*: SamplerParams
    stop*: seq[string]           # chaînes qui arrêtent la génération.
    onToken*: TokenCallback      # affichage en flux (optionnel).

  Reply* = object
    # Résultat d'une génération.
    text*: string                # texte final (nettoyé selon le format).
    raw*: string                 # texte brut produit par le modèle.
    files*: seq[string]          # fichiers créés (image, audio, fichier).
    json*: JsonNode              # objet JSON (format ofJson).
    promptTokens*: int
    completionTokens*: int
    stopReason*: StopReason
    seconds*: float
    tokensPerSecond*: float

  Chat* = ref object
    # Conversation : un modèle, un contexte KV et un historique.
    model*: LlmModel
    ctx*: LlmContext
    system*: string
    history*: seq[Message]
    templ*: ChatTemplateKind
    options*: GenOptions
    stripThinking*: bool         # retire les blocs <think>...</think>
    lang*: string                # "fr" ou "en" (instructions de format, voix)
    imageSize*: int              # taille des images produites (pixels)
    jsonRetries*: int
    voicePitch*: float
    sampler: Sampler

  ChatError* = object of CatchableError

proc defaultOptions*(): GenOptions =
  GenOptions(maxTokens: 512, sampling: defaultSampling())

#
# Modèles de conversation
#

proc detectTemplate*(tok: Tokenizer): ChatTemplateKind =
  # Devine le format de conversation à partir du modèle de chat GGUF ou du vocabulaire.
  let t = tok.chatTemplate
  if "<|start_header_id|>" in t or tok.tokenId("<|start_header_id|>") >= 0: return tplLlama3
  if "<|im_start|>" in t or tok.tokenId("<|im_start|>") >= 0: return tplChatML
  if "<start_of_turn>" in t or tok.tokenId("<start_of_turn>") >= 0: return tplGemma
  if ("<|user|>" in t and "<|assistant|>" in t) or tok.tokenId("<|assistant|>") >= 0: return tplPhi3
  if "<<SYS>>" in t: return tplLlama2
  if "[INST]" in t or tok.tokenId("[INST]") >= 0: return tplMistral
  tplRaw

type Segment = tuple[text: string, special: bool]

proc segments(kind: ChatTemplateKind; msgs: openArray[Message]; addGen: bool;
              bos: string): seq[Segment] =
  # Produit la suite de segments (balises du gabarit / contenu utilisateur).
  template sp(s: string) = result.add (s, true)
  template tx(s: string) = result.add (s, false)
  case kind
  of tplLlama3, tplAuto:
    sp "<|begin_of_text|>"
    for m in msgs:
      sp "<|start_header_id|>" & $m.role & "<|end_header_id|>\n\n"
      tx m.content
      sp "<|eot_id|>"
    if addGen: sp "<|start_header_id|>assistant<|end_header_id|>\n\n"
  of tplChatML:
    for m in msgs:
      sp "<|im_start|>" & $m.role & "\n"
      tx m.content
      sp "<|im_end|>\n"
    if addGen: sp "<|im_start|>assistant\n"
  of tplPhi3:
    for m in msgs:
      sp "<|" & $m.role & "|>\n"
      tx m.content
      sp "<|end|>\n"
    if addGen: sp "<|assistant|>\n"
  of tplGemma:
    sp bos
    var sys = ""
    for m in msgs:
      if m.role == roleSystem: sys = m.content; continue
      let r = if m.role == roleAssistant: "model" else: "user"
      sp "<start_of_turn>" & r & "\n"
      if sys.len > 0 and m.role == roleUser:
        tx sys & "\n\n"; sys = ""
      tx m.content
      sp "<end_of_turn>\n"
    if addGen: sp "<start_of_turn>model\n"
  of tplMistral, tplLlama2:
    var sys = ""
    var first = true
    for m in msgs:
      case m.role
      of roleSystem: sys = m.content
      of roleUser:
        sp bos & "[INST]"
        var c = " "
        if sys.len > 0 and first:
          c &= (if kind == tplLlama2: "<<SYS>>\n" & sys & "\n<</SYS>>\n\n" else: sys & "\n\n")
        tx c & m.content
        sp " [/INST]"
        first = false
      of roleAssistant:
        tx " " & m.content
        sp "</s>"
  of tplRaw:
    for m in msgs:
      case m.role
      of roleSystem: tx m.content & "\n\n"
      of roleUser: tx "Utilisateur : " & m.content & "\n"
      of roleAssistant: tx "Assistant : " & m.content & "\n"
    if addGen: tx "Assistant :"

proc renderPrompt*(kind: ChatTemplateKind; msgs: openArray[Message];
                   addGenerationPrompt = true; bos = "<s>"): string =
  # Texte complet du prompt (utile pour inspecter ou préparer des données).
  for s in segments(kind, msgs, addGenerationPrompt, bos): result.add s.text

proc encodeMessages*(tok: Tokenizer; kind: ChatTemplateKind; msgs: openArray[Message];
                     addGenerationPrompt = true): seq[int] =
  # Tokenise une conversation. Les balises du gabarit sont reconnues comme
  # tokens spéciaux ; le contenu des messages ne l'est jamais (sécurité).
  let bos = if tok.bosId >= 0: tok.tokens[tok.bosId] else: ""
  let k = if kind == tplAuto: detectTemplate(tok) else: kind
  if k == tplRaw and tok.addBos and tok.bosId >= 0: result.add tok.bosId
  for s in segments(k, msgs, addGenerationPrompt, bos):
    if s.text.len == 0: continue
    result.add tok.encode(s.text, addBos = false, parseSpecial = s.special)

proc stopStringsFor*(kind: ChatTemplateKind): seq[string] =
  case kind
  of tplRaw: @["\nUtilisateur :", "\nUtilisateur:", "\nUser:"]
  of tplMistral, tplLlama2: @["[INST]"]
  else: @[]

#
# Génération
#

proc validPrefixLen(s: string): int =
  # Longueur du plus long préfixe UTF-8 complet.
  var i = 0
  while i < s.len:
    let c = ord(s[i])
    let n = if c < 0x80: 1 elif c shr 5 == 6: 2 elif c shr 4 == 14: 3 elif c shr 3 == 30: 4 else: 1
    if i + n > s.len: break
    i += n
  i

proc generate*(ctx: LlmContext; prompt: seq[int]; opts: GenOptions;
               smp: Sampler = nil): Reply =
  # Génère une suite à partir des tokens `prompt` (aucun gabarit appliqué).
  # Le cache est réutilisé si `prompt` prolonge le contenu déjà évalué.
  let t0 = epochTime()
  let tok = ctx.model.tokenizer
  let s = if smp != nil: smp else: newSampler(opts.sampling)
  if prompt.len >= ctx.nCtx:
    raise newException(ChatError, "le prompt (" & $prompt.len &
      " tokens) dépasse le contexte (" & $ctx.nCtx & ")")
  result.promptTokens = prompt.len
  var logits = ctx.evalPrompt(prompt)
  var pending = ""
  var emitted = 0
  result.stopReason = srMaxTokens
  let maxT = if opts.maxTokens <= 0: ctx.nCtx else: opts.maxTokens
  var outToks: seq[int]
  for step in 0 ..< maxT:
    let id = s.sample(logits)
    if id in tok.eogIds:
      result.stopReason = srEndOfText; break
    s.accept id
    outToks.add id
    result.raw.add tok.tokenToPiece(id)
    # arrêt sur chaîne.
    var stopHit = false
    for st in opts.stop:
      if st.len > 0 and result.raw.endsWith(st):
        result.raw.setLen(result.raw.len - st.len)
        stopHit = true
    # flux UTF-8 sûr.
    if opts.onToken != nil:
      let avail = result.raw.len
      if avail > emitted:
        pending = result.raw[emitted ..< avail]
        # retient ce qui pourrait être le début d'une chaîne d'arrêt.
        var hold = 0
        if not stopHit:
          for st in opts.stop:
            for k in countdown(min(st.len - 1, pending.len), 1):
              if result.raw.endsWith(st[0 ..< k]): hold = max(hold, k); break
        let n = validPrefixLen(pending[0 ..< pending.len - hold])
        if n > 0:
          if not opts.onToken(pending[0 ..< n]):
            emitted += n
            result.stopReason = srCallback
            break
          emitted += n
    if stopHit:
      result.stopReason = srStopString; break
    if ctx.nPast >= ctx.nCtx:
      result.stopReason = srContextFull; break
    logits = ctx.eval([id])
  if opts.onToken != nil and emitted < result.raw.len and result.stopReason != srCallback:
    discard opts.onToken(result.raw[emitted .. ^1])
  result.completionTokens = outToks.len
  # retire un éventuel caractère UTF-8 incomplet en fin de texte.
  result.raw.setLen(validPrefixLen(result.raw))
  result.text = result.raw
  result.seconds = epochTime() - t0
  if result.seconds > 0: result.tokensPerSecond = outToks.len.float / result.seconds

proc complete*(ctx: LlmContext; text: string; opts = defaultOptions()): Reply =
  # Complétion brute d'un texte (sans gabarit de conversation).
  let tok = ctx.model.tokenizer
  ctx.generate(tok.encode(text, addBos = tok.addBos), opts)

#
# Conversation
#

proc newChat*(m: LlmModel; system = ""; nCtx = 4096; templ = tplAuto;
              sampling = defaultSampling(); maxTokens = 512; lang = "fr"): Chat =
  # Crée une conversation. `system` définit le contexte / le rôle du modèle.
  result = Chat(model: m, ctx: newContext(m, nCtx), system: system,
                templ: (if templ == tplAuto: detectTemplate(m.tokenizer) else: templ),
                lang: lang, imageSize: 512, jsonRetries: 2, voicePitch: 115,
                stripThinking: true)
  result.options = GenOptions(maxTokens: maxTokens, sampling: sampling)
  result.sampler = newSampler(sampling)

proc setSampling*(c: Chat; p: SamplerParams) =
  c.options.sampling = p
  c.sampler = newSampler(p)

proc messages*(c: Chat): seq[Message] =
  # Messages envoyés au modèle (prompt système + historique).
  if c.system.len > 0: result.add Message(role: roleSystem, content: c.system)
  result.add c.history

proc reset*(c: Chat) =
  # Efface l'historique (garde le prompt système).
  c.history.setLen(0)

proc undo*(c: Chat) =
  # Retire le dernier échange question/réponse.
  while c.history.len > 0 and c.history[^1].role == roleAssistant: c.history.setLen(c.history.len - 1)
  if c.history.len > 0 and c.history[^1].role == roleUser: c.history.setLen(c.history.len - 1)

proc add*(c: Chat; role: Role; content: string) =
  # Ajoute un message à l'historique sans rien générer (few-shot, reprise...).
  c.history.add Message(role: role, content: content)

proc promptTokens*(c: Chat; addGen = true): seq[int] =
  encodeMessages(c.model.tokenizer, c.templ, c.messages, addGen)

proc fitContext(c: Chat; reserve: int): seq[int] =
  # Retire les plus anciens échanges si la conversation dépasse le contexte.
  result = c.promptTokens()
  while result.len + reserve > c.ctx.nCtx and c.history.len > 1:
    # supprime le plus ancien message (et sa réponse).
    c.history.delete(0)
    if c.history.len > 1 and c.history[0].role == roleAssistant: c.history.delete(0)
    result = c.promptTokens()
  if result.len + 1 >= c.ctx.nCtx:
    raise newException(ChatError, "message trop long pour le contexte (" & $result.len &
      " tokens, contexte " & $c.ctx.nCtx & ")")

proc removeThinking(s: string): string =
  result = s
  while true:
    let a = result.find("<think>")
    if a < 0: break
    let b = result.find("</think>", a)
    if b < 0: result = result[0 ..< a]; break
    result = result[0 ..< a] & result[b + 8 .. ^1]
  result = result.strip

proc formatInstruction(c: Chat; format: OutputFormat; schema, fileName: string): string =
  let fr = c.lang == "fr"
  case format
  of ofText: ""
  of ofMarkdown:
    if fr: "\n\n(Mets en forme ta réponse en Markdown.)"
    else: "\n\n(Format your answer in Markdown.)"
  of ofJson:
    (if fr: "\n\nRéponds UNIQUEMENT avec un JSON valide, sans aucun texte autour."
     else: "\n\nAnswer ONLY with valid JSON, no other text.") &
    (if schema.len > 0: (if fr: " Structure attendue : " else: " Expected structure: ") & schema else: "")
  of ofImage:
    let s = $c.imageSize
    if fr: "\n\nRéponds UNIQUEMENT avec le code SVG complet de l'image (de <svg à </svg>), " &
           "avec viewBox=\"0 0 " & s & " " & s & "\", en utilisant des formes simples " &
           "(rect, circle, ellipse, polygon, path, line) et des couleurs de remplissage. Pas d'explication."
    else: "\n\nAnswer ONLY with the complete SVG code (from <svg to </svg>), viewBox=\"0 0 " &
          s & " " & s & "\", using simple shapes and fill colors. No explanation."
  of ofAudio:
    if fr: "\n\n(Ta réponse sera lue à voix haute : écris des phrases simples, sans Markdown, " &
           "sans listes, sans symboles ni émojis.)"
    else: "\n\n(Your answer will be read aloud: plain short sentences, no Markdown, lists or symbols.)"
  of ofFile:
    if fr: "\n\nRéponds UNIQUEMENT avec le contenu brut du fichier " & fileName &
           ", sans explication et sans balises Markdown."
    else: "\n\nAnswer ONLY with the raw content of the file " & fileName &
          ", no explanation and no Markdown fences."

proc stripFences*(s: string): string =
  # Retire les balises ```...``` entourant un bloc de code.
  var t = s.strip
  let a = t.find("```")
  if a >= 0:
    let nl = t.find('\n', a)
    let b = t.find("```", max(a + 3, nl))
    if nl >= 0 and b > nl: return t[nl + 1 ..< b].strip(leading = false)
    if nl >= 0: return t[nl + 1 .. ^1]
  t

proc extractJson*(s: string): JsonNode =
  # Extrait et analyse le premier objet ou tableau JSON d'un texte
  # (les valeurs isolées comme `42` ne sont pas acceptées).
  # Lève `JsonParsingError` si aucun JSON valide n'est trouvé.
  let t = stripFences(s)
  try:
    let j = parseJson(t)
    if j.kind in {JObject, JArray}: return j
  except CatchableError: discard
  for opener in ['{', '[']:
    let closer = if opener == '{': '}' else: ']'
    let a = t.find(opener)
    if a < 0: continue
    var depth = 0
    var inStr = false
    var esc = false
    for i in a ..< t.len:
      let ch = t[i]
      if inStr:
        if esc: esc = false
        elif ch == '\\': esc = true
        elif ch == '"': inStr = false
      elif ch == '"': inStr = true
      elif ch == opener: inc depth
      elif ch == closer:
        dec depth
        if depth == 0:
          try:
            let j = parseJson(t[a .. i])
            if j.kind in {JObject, JArray}: return j
          except CatchableError: discard
          break
  raise newException(JsonParsingError, "aucun JSON valide dans la réponse")

proc defaultOutPath(format: OutputFormat): string =
  let stamp = format(now(), "yyyyMMdd'_'HHmmss")
  case format
  of ofImage: "image_" & stamp & ".svg"
  of ofAudio: "reponse_" & stamp & ".wav"
  of ofFile: "fichier_" & stamp & ".txt"
  else: ""

proc runTurn(c: Chat; opts: GenOptions): Reply =
  let reserve = min(opts.maxTokens, c.ctx.nCtx div 2)
  let toks = c.fitContext(reserve)
  var o = opts
  for s in stopStringsFor(c.templ): o.stop.add s
  result = c.ctx.generate(toks, o, c.sampler)
  if c.stripThinking: result.text = removeThinking(result.raw)
  else: result.text = result.raw.strip
  c.history.add Message(role: roleAssistant, content: result.raw.strip)

proc ask*(c: Chat; prompt: string; attachments: seq[Attachment] = @[];
          format = ofText; outPath = ""; schema = ""; onToken: TokenCallback = nil;
          maxTokens = 0): Reply =
  # Pose une question et retourne la réponse.
  #
  # * `attachments` : pièces jointes (voir `attach`) ajoutées au message ;
  # * `format` : forme de la réponse (texte, markdown, json, image, audio, fichier) ;
  # * `outPath` : fichier de sortie pour image / audio / fichier ;
  # * `schema` : description de la structure JSON attendue (format json) ;
  # * `onToken` : rappel recevant le texte au fil de la génération.
  var content = prompt
  for a in attachments:
    content.add "\n\n" & a.toPrompt()
  let fileName = if outPath.len > 0: outPath.extractFilename else: "demandé"
  content.add c.formatInstruction(format, schema, fileName)
  c.history.add Message(role: roleUser, content: content)
  var opts = c.options
  if maxTokens > 0: opts.maxTokens = maxTokens
  if onToken != nil: opts.onToken = onToken
  if format in {ofImage, ofFile} and opts.maxTokens < 1024 and maxTokens == 0:
    opts.maxTokens = 1536
  result = c.runTurn(opts)
  case format
  of ofText, ofMarkdown: discard
  of ofJson:
    var tries = 0
    while true:
      try:
        result.json = extractJson(result.text)
        result.text = $result.json
        break
      except CatchableError:
        if tries >= c.jsonRetries:
          raise newException(ChatError, "le modèle n'a pas produit de JSON valide : " &
            result.text[0 ..< min(200, result.text.len)])
        inc tries
        # nouvelle tentative, plus déterministe.
        c.history.setLen(c.history.len - 1)
        var o2 = opts
        o2.sampling.temperature = max(0.0, o2.sampling.temperature * 0.5)
        o2.onToken = nil
        result = c.runTurn(o2)
  of ofImage:
    let svg = extractSvg(result.text)
    if svg.len == 0:
      raise newException(ChatError, "aucun code SVG dans la réponse")
    let p = if outPath.len > 0: outPath else: defaultOutPath(ofImage)
    let base = p.changeFileExt("")
    writeFile(base & ".svg", svg)
    result.files.add base & ".svg"
    let img = renderSvg(svg, c.imageSize, 0)
    img.writeBmp(base & ".bmp")
    result.files.add base & ".bmp"
    result.text = svg
  of ofAudio:
    let p = if outPath.len > 0: outPath else: defaultOutPath(ofAudio)
    var clean = result.text.multiReplace(("*", ""), ("#", ""), ("`", ""), ("_", " "))
    let a = speak(clean, c.lang, c.voicePitch)
    a.writeWav(p)
    result.files.add p
  of ofFile:
    let p = if outPath.len > 0: outPath else: defaultOutPath(ofFile)
    result.text = stripFences(result.text)
    writeFile(p, result.text & "\n")
    result.files.add p

proc askJson*(c: Chat; prompt: string; schema = ""): JsonNode =
  # Pose une question et retourne directement l'objet JSON de la réponse.
  c.ask(prompt, format = ofJson, schema = schema).json

proc regenerate*(c: Chat; onToken: TokenCallback = nil): Reply =
  # Régénère la dernière réponse (nouveau tirage).
  if c.history.len == 0 or c.history[^1].role != roleAssistant:
    raise newException(ChatError, "aucune réponse à régénérer")
  c.history.setLen(c.history.len - 1)
  var opts = c.options
  opts.onToken = onToken
  c.runTurn(opts)

proc continueReply*(c: Chat; maxTokens = 256; onToken: TokenCallback = nil): Reply =
  # Prolonge la dernière réponse (si elle a été coupée par la limite de tokens).
  if c.history.len == 0 or c.history[^1].role != roleAssistant:
    raise newException(ChatError, "aucune réponse à prolonger")
  let last = c.history[^1].content
  var toks = encodeMessages(c.model.tokenizer, c.templ, c.messages[0 ..< ^1], true)
  toks.add c.model.tokenizer.encode(last, parseSpecial = false)
  var opts = c.options
  opts.maxTokens = maxTokens
  opts.onToken = onToken
  result = c.ctx.generate(toks, opts, c.sampler)
  c.history[^1].content = last & result.raw
  result.text = c.history[^1].content

#
# Sauvegarde
#

proc toJson*(c: Chat): JsonNode =
  result = %*{"system": c.system, "template": $c.templ, "messages": []}
  for m in c.history:
    result["messages"].add %*{"role": $m.role, "content": m.content}

proc saveHistory*(c: Chat; path: string) =
  # Enregistre la conversation (JSON).
  writeFile(path, c.toJson.pretty)

proc loadHistory*(c: Chat; path: string) =
  # Recharge une conversation enregistrée par `saveHistory`.
  let j = parseFile(path)
  c.system = j{"system"}.getStr(c.system)
  c.history.setLen(0)
  for m in j{"messages"}:
    let r = case m["role"].getStr
      of "system": roleSystem
      of "assistant": roleAssistant
      else: roleUser
    c.history.add Message(role: r, content: m["content"].getStr)

proc stats*(c: Chat): string =
  # Résumé de l'état du contexte.
  "contexte : " & $c.ctx.nPast & "/" & $c.ctx.nCtx & " tokens, " & $c.history.len &
  " messages, vitesse moyenne " & formatFloat(c.ctx.tokensPerSecond, ffDecimal, 1) & " tok/s"
