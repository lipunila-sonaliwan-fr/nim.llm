# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026 (adaptation)
# Portions dérivées de ggml / llama.cpp
# The ggml authors (© 2023-2026) - LICENSES/MIT-ggml.txt
# Correspondance octets ↔ caractères Unicode du BPE niveau octet (fonction bytes_to_unicode de GPT-2)
# OpenAI (© 2019) - LICENSES/MIT-OpenAI-GPT2.txt
# nimllm/tokenizer — tokeniseurs compatibles llama.cpp :
#
# * **BPE** niveau octet (GPT-2, Llama 3, Qwen 2...) : `tokenizer.ggml.model = "gpt2"`
# * **SPM** (SentencePiece, Llama 2, Mistral...) : `tokenizer.ggml.model = "llama"`
#
# Le vocabulaire est lu dans les métadonnées GGUF ; un tokeniseur peut aussi
# être construit (ou entraîné, voir `tokentrain`) puis enregistré dans un GGUF.

import std/[tables, strutils, unicode, heapqueue, sets, algorithm]
import gguf, unicode_tables
export sets

type
  TokenizerKind* = enum
    tkBpe = "gpt2"               # BPE niveau octet.
    tkSpm = "llama"              # SentencePiece.

  TokenType* = enum
    ttUndefined = 0, ttNormal = 1, ttUnknown = 2, ttControl = 3,
    ttUserDefined = 4, ttUnused = 5, ttByte = 6

  PreTokenizer* = enum
    ptLlama3 = "llama-bpe"       # regex Llama 3 (nombres par groupes de 3).
    ptQwen2 = "qwen2"            # comme Llama 3 mais chiffres un par un.
    ptGpt2 = "gpt2"              # regex GPT-2 d'origine.

  Tokenizer* = ref object
    # Tokeniseur : conversion texte <-> identifiants de tokens.
    kind*: TokenizerKind
    pre*: PreTokenizer
    tokens*: seq[string]              # texte brut de chaque token (forme interne).
    scores*: seq[float32]
    types*: seq[TokenType]
    merges*: seq[string]              # fusions BPE "a b" par ordre de priorité.
    tokenToId*: Table[string, int]
    mergeRank: Table[string, int]
    specials: seq[int]                # tokens spéciaux, triés par longueur décroissante.
    bosId*, eosId*, unkId*, padId*: int
    eogIds*: HashSet[int]             # tokens de fin de génération (eos, eot...)
    addBos*: bool
    addSpacePrefix*: bool
    ignoreMerges*: bool
    chatTemplate*: string
    byteTokens: array[256, int]

#
# Unicode
#

proc inRanges(c: int32; r: openArray[(int32, int32)]): bool =
  var lo = 0
  var hi = r.high
  while lo <= hi:
    let m = (lo + hi) div 2
    if c < r[m][0]: hi = m - 1
    elif c > r[m][1]: lo = m + 1
    else: return true
  false

proc isLetter*(r: Rune): bool =
  let c = int32(r)
  if c < 128: return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)
  inRanges(c, letterRanges)

proc isNumber*(r: Rune): bool =
  let c = int32(r)
  if c < 128: return c >= 48 and c <= 57
  inRanges(c, numberRanges)

proc isSpace*(r: Rune): bool =
  let c = int32(r)
  case c
  of 9..13, 32, 0x85, 0xA0, 0x1680, 0x2000..0x200A, 0x2028, 0x2029, 0x202F,
     0x205F, 0x3000: true
  else: false

proc isNewline(r: Rune): bool = int32(r) == 10 or int32(r) == 13

#
# Pré-tokenisation (équivalent des expressions régulières de llama.cpp)
#

proc preTokenize*(text: string; mode: PreTokenizer): seq[string] =
  # Découpe le texte en « mots » avant l'application du BPE.
  let rs = toRunes(text)
  let n = rs.len
  var i = 0
  template L(k: int): bool = k < n and isLetter(rs[k])
  template N(k: int): bool = k < n and isNumber(rs[k])
  template S(k: int): bool = k < n and isSpace(rs[k])
  template emit(a, b: int) =
    result.add $rs[a ..< b]
    i = b
  while i < n:
    let c = rs[i]
    # contractions anglaises.
    if int32(c) == ord('\'') and i + 1 < n:
      let c1 = toLower(rs[i+1])
      let c2 = if i + 2 < n: toLower(rs[i+2]) else: Rune(0)
      if c1 == Rune('s') or c1 == Rune('t') or c1 == Rune('m') or c1 == Rune('d'):
        emit(i, i + 2); continue
      if (c1 == Rune('r') and c2 == Rune('e')) or (c1 == Rune('v') and c2 == Rune('e')) or
         (c1 == Rune('l') and c2 == Rune('l')):
        emit(i, i + 3); continue
    if mode == ptGpt2:
      # ' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'
      let start = i
      var j = i
      if int32(rs[j]) == 32 and j + 1 < n and not isSpace(rs[j+1]): inc j
      if L(j):
        while L(j): inc j
        emit(start, j); continue
      if N(j):
        while N(j): inc j
        emit(start, j); continue
      if j < n and not S(j) and not L(j) and not N(j):
        while j < n and not S(j) and not L(j) and not N(j): inc j
        emit(start, j); continue
    else:
      # [^\r\n\p{L}\p{N}]?\p{L}+
      if L(i):
        var j = i
        while L(j): inc j
        emit(i, j); continue
      if not isNewline(c) and not isNumber(c) and L(i + 1):
        var j = i + 1
        while L(j): inc j
        emit(i, j); continue
      # \p{N}{1,3}  (ou \p{N} pour qwen2)
      if N(i):
        let maxD = if mode == ptQwen2: 1 else: 3
        var j = i
        while N(j) and j - i < maxD: inc j
        emit(i, j); continue
      # ' ?[^\s\p{L}\p{N}]+[\r\n]*'
      block punct:
        var j = i
        if int32(rs[j]) == 32: inc j
        if j < n and not S(j) and not L(j) and not N(j):
          while j < n and not S(j) and not L(j) and not N(j): inc j
          while j < n and isNewline(rs[j]): inc j
          emit(i, j)
          break punct
        # \s*[\r\n]+
        if S(i):
          var j = i
          var lastNl = -1
          while S(j):
            if isNewline(rs[j]): lastNl = j
            inc j
          if lastNl >= 0:
            emit(i, lastNl + 1); break punct
          # \s+(?!\S)  puis 7. \s+
          if j >= n or j - i == 1:
            emit(i, j)
          else:
            emit(i, j - 1)
          break punct
        # sécurité : caractère isolé.
        emit(i, i + 1)
      continue
    # chemin GPT-2 : espaces.
    if S(i):
      var j = i
      while S(j): inc j
      if j >= n or j - i == 1: emit(i, j)
      else: emit(i, j - 1)
      continue
    emit(i, i + 1)

#
# Correspondance octets <-> caractères (GPT-2)
#

var byteToUni*: array[256, string]
var uniToByte*: Table[int32, uint8]

proc initByteMap() =
  var bs: seq[int]
  for b in ord('!') .. ord('~'): bs.add b
  for b in 0xA1 .. 0xAC: bs.add b
  for b in 0xAE .. 0xFF: bs.add b
  var cs = bs
  var k = 0
  for b in 0 .. 255:
    if b notin bs:
      bs.add b
      cs.add 256 + k
      inc k
  for i in 0 ..< bs.len:
    byteToUni[bs[i]] = $Rune(cs[i])
    uniToByte[int32(cs[i])] = uint8(bs[i])

initByteMap()

proc byteEncode*(s: string): string =
  # Transforme des octets UTF-8 en caractères « imprimables » (convention GPT-2).
  for ch in s: result.add byteToUni[ord(ch)]

proc byteDecode*(s: string): string =
  for r in s.runes:
    let c = int32(r)
    if c in uniToByte: result.add char(uniToByte[c])
    else: result.add $r

#
# Construction
#

proc rebuild*(t: Tokenizer) =
  # Recalcule les index internes après modification du vocabulaire.
  t.tokenToId = initTable[string, int]()
  for i, s in t.tokens:
    if s notin t.tokenToId: t.tokenToId[s] = i
  t.mergeRank = initTable[string, int]()
  for i, m in t.merges: t.mergeRank[m] = i
  if t.types.len < t.tokens.len:
    for i in t.types.len ..< t.tokens.len: t.types.add ttNormal
  if t.scores.len < t.tokens.len:
    for i in t.scores.len ..< t.tokens.len: t.scores.add 0
  t.specials = @[]
  for i, ty in t.types:
    if ty in {ttControl, ttUserDefined, ttUnknown} and t.tokens[i].len > 0:
      t.specials.add i
  t.specials.sort(proc(a, b: int): int = cmp(t.tokens[b].len, t.tokens[a].len))
  for b in 0 .. 255:
    t.byteTokens[b] = -1
    if t.kind == tkSpm:
      let k = "<0x" & toHex(b, 2) & ">"
      if k in t.tokenToId: t.byteTokens[b] = t.tokenToId[k]
    else:
      let k = byteToUni[b]
      if k in t.tokenToId: t.byteTokens[b] = t.tokenToId[k]
  # tokens de fin de génération.
  for name in ["<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>",
               "<|endoftext|>", "</s>", "<|end_of_text|>", "<eos>"]:
    if name in t.tokenToId and t.types[t.tokenToId[name]] in {ttControl, ttUserDefined}:
      t.eogIds.incl t.tokenToId[name]
  if t.eosId >= 0: t.eogIds.incl t.eosId

proc tokenizerFromGguf*(g: GgufFile): Tokenizer =
  # Construit le tokeniseur décrit dans les métadonnées d'un fichier GGUF.
  result = Tokenizer(bosId: -1, eosId: -1, unkId: -1, padId: -1)
  let model = g.getStr("tokenizer.ggml.model", "gpt2")
  case model
  of "gpt2": result.kind = tkBpe
  of "llama": result.kind = tkSpm
  else:
    raise newException(ValueError, "modèle de tokeniseur non pris en charge : " & model)
  result.tokens = g.getStrArray("tokenizer.ggml.tokens")
  if result.tokens.len == 0:
    raise newException(ValueError, "le fichier ne contient pas de vocabulaire")
  result.scores = g.getFloatArray("tokenizer.ggml.scores")
  for v in g.getIntArray("tokenizer.ggml.token_type"):
    result.types.add(if v in 0..6: TokenType(v) else: ttNormal)
  result.merges = g.getStrArray("tokenizer.ggml.merges")
  result.bosId = g.getInt("tokenizer.ggml.bos_token_id", -1)
  result.eosId = g.getInt("tokenizer.ggml.eos_token_id", -1)
  result.unkId = g.getInt("tokenizer.ggml.unknown_token_id", -1)
  result.padId = g.getInt("tokenizer.ggml.padding_token_id", -1)
  let pre = g.getStr("tokenizer.ggml.pre", "default")
  case pre
  of "llama-bpe", "llama3", "smaug-bpe", "dbrx", "tekken":
    result.pre = ptLlama3; result.ignoreMerges = true
  of "qwen2", "deepseek-r1-qwen", "qwen35":
    result.pre = ptQwen2
  of "gpt2", "default", "gpt-2":
    result.pre = (if model == "gpt2" and pre == "default": ptGpt2 else: ptGpt2)
  else:
    result.pre = ptLlama3
  if g.has("tokenizer.ggml.ignore_merges"):
    result.ignoreMerges = g.getBool("tokenizer.ggml.ignore_merges")
  result.addBos = g.getBool("tokenizer.ggml.add_bos_token",
                            result.kind == tkSpm or result.pre == ptLlama3)
  result.addSpacePrefix = g.getBool("tokenizer.ggml.add_space_prefix", result.kind == tkSpm)
  result.chatTemplate = g.getStr("tokenizer.chat_template")
  result.rebuild()

proc writeToGguf*(t: Tokenizer; w: GgufWriter) =
  # Enregistre le tokeniseur dans les métadonnées d'un fichier GGUF.
  w.setKV("tokenizer.ggml.model", gStr($t.kind))
  if t.kind == tkBpe: w.setKV("tokenizer.ggml.pre", gStr($t.pre))
  w.setKV("tokenizer.ggml.tokens", gArrStr(t.tokens))
  var tt: seq[int]
  for x in t.types: tt.add ord(x)
  w.setKV("tokenizer.ggml.token_type", gArrI32(tt))
  if t.kind == tkSpm: w.setKV("tokenizer.ggml.scores", gArrF32(t.scores))
  if t.merges.len > 0: w.setKV("tokenizer.ggml.merges", gArrStr(t.merges))
  if t.bosId >= 0: w.setKV("tokenizer.ggml.bos_token_id", gU32(t.bosId))
  if t.eosId >= 0: w.setKV("tokenizer.ggml.eos_token_id", gU32(t.eosId))
  if t.unkId >= 0: w.setKV("tokenizer.ggml.unknown_token_id", gU32(t.unkId))
  if t.padId >= 0: w.setKV("tokenizer.ggml.padding_token_id", gU32(t.padId))
  w.setKV("tokenizer.ggml.add_bos_token", gBool(t.addBos))
  w.setKV("tokenizer.ggml.add_space_prefix", gBool(t.addSpacePrefix))
  if t.chatTemplate.len > 0: w.setKV("tokenizer.chat_template", gStr(t.chatTemplate))

proc vocabSize*(t: Tokenizer): int = t.tokens.len

proc tokenId*(t: Tokenizer; text: string): int =
  # Identifiant d'un token par son texte exact (-1 si absent).
  t.tokenToId.getOrDefault(text, -1)

proc isSpecial*(t: Tokenizer; id: int): bool =
  id >= 0 and id < t.types.len and t.types[id] in {ttControl, ttUserDefined, ttUnknown}

#
# Encodage BPE
#

type
  Sym = object
    text: string
    prev, next: int
    alive: bool
  BpeBigram = object
    left, right, rank: int
    text: string
  SpmBigram = object
    left, right: int
    score: float32
    size: int

proc `<`(a, b: BpeBigram): bool =
  a.rank < b.rank or (a.rank == b.rank and a.left < b.left)

proc `<`(a, b: SpmBigram): bool =
  a.score > b.score or (a.score == b.score and a.left < b.left)

proc bpeWord(t: Tokenizer; word: string; output: var seq[int]) =
  if t.ignoreMerges and word in t.tokenToId:
    output.add t.tokenToId[word]; return
  var syms: seq[Sym]
  for r in word.runes:
    syms.add Sym(text: $r, prev: syms.len - 1, next: syms.len + 1, alive: true)
  if syms.len == 0: return
  syms[^1].next = -1
  var q = initHeapQueue[BpeBigram]()
  proc tryAdd(q: var HeapQueue[BpeBigram]; l, r: int) =
    if l < 0 or r < 0: return
    let key = syms[l].text & " " & syms[r].text
    let rank = t.mergeRank.getOrDefault(key, -1)
    if rank < 0: return
    q.push BpeBigram(left: l, right: r, rank: rank, text: syms[l].text & syms[r].text)
  for i in 1 ..< syms.len: tryAdd(q, i - 1, i)
  while q.len > 0:
    let bg = q.pop()
    let l = bg.left
    let r = bg.right
    if not syms[l].alive or not syms[r].alive: continue
    if syms[l].text & syms[r].text != bg.text: continue
    syms[l].text = bg.text
    syms[r].alive = false
    syms[l].next = syms[r].next
    if syms[r].next >= 0: syms[syms[r].next].prev = l
    tryAdd(q, syms[l].prev, l)
    tryAdd(q, l, syms[l].next)
  var i = 0
  while i != -1:
    let s = syms[i]
    let id = t.tokenToId.getOrDefault(s.text, -1)
    if id >= 0:
      output.add id
    else:
      for ch in byteDecode(s.text):
        let b = t.byteTokens[ord(ch)]
        if b >= 0: output.add b
        elif t.unkId >= 0: output.add t.unkId
    i = s.next

proc spmText(t: Tokenizer; text: string; output: var seq[int]) =
  # découpage en caractères UTF-8.
  var syms: seq[Sym]
  var offs = 0
  while offs < text.len:
    let rl = min(runeLenAt(text, offs), text.len - offs)
    syms.add Sym(text: text[offs ..< offs + rl], prev: syms.len - 1, next: syms.len + 1, alive: true)
    offs += rl
  if syms.len == 0: return
  syms[^1].next = -1
  var q = initHeapQueue[SpmBigram]()
  proc tryAdd(q: var HeapQueue[SpmBigram]; l, r: int) =
    if l < 0 or r < 0: return
    let s = syms[l].text & syms[r].text
    let id = t.tokenToId.getOrDefault(s, -1)
    if id < 0: return
    q.push SpmBigram(left: l, right: r, score: t.scores[id], size: s.len)
  for i in 1 ..< syms.len: tryAdd(q, i - 1, i)
  # fusion des paires de meilleur score.
  while q.len > 0:
    let bg = q.pop()
    let l = bg.left
    let r = bg.right
    if not syms[l].alive or not syms[r].alive: continue
    if syms[l].text.len + syms[r].text.len != bg.size: continue
    syms[l].text = syms[l].text & syms[r].text
    syms[r].alive = false
    syms[l].next = syms[r].next
    if syms[r].next >= 0: syms[syms[r].next].prev = l
    tryAdd(q, syms[l].prev, l)
    tryAdd(q, l, syms[l].next)
  # émission : token connu, sinon repli sur les octets <0xXX>.
  var i = 0
  while i != -1:
    let s = syms[i].text
    let id = t.tokenToId.getOrDefault(s, -1)
    if id >= 0:
      output.add id
    else:
      for ch in s:
        let b = t.byteTokens[ord(ch)]
        if b >= 0: output.add b
        elif t.unkId >= 0: output.add t.unkId
    i = syms[i].next

proc encode*(t: Tokenizer; text: string; addBos = false; parseSpecial = true): seq[int] =
  # Encode `text` en identifiants de tokens.
  # * `addBos` : ajoute le token de début de séquence si le modèle en a un ;
  # * `parseSpecial` : reconnaît les tokens spéciaux écrits dans le texte
  #   (ex. `<|eot_id|>`).
  if addBos and t.bosId >= 0: result.add t.bosId
  # découpage autour des tokens spéciaux.
  type Frag = object
    isTok: bool
    tok: int
    text: string
  var frags: seq[Frag]
  if parseSpecial and t.specials.len > 0:
    var cur = ""
    var i = 0
    while i < text.len:
      var matched = -1
      for id in t.specials:
        let s = t.tokens[id]
        if s[0] == text[i] and text.continuesWith(s, i):
          matched = id; break
      if matched >= 0:
        if cur.len > 0: frags.add Frag(text: cur); cur = ""
        frags.add Frag(isTok: true, tok: matched)
        i += t.tokens[matched].len
      else:
        cur.add text[i]; inc i
    if cur.len > 0: frags.add Frag(text: cur)
  else:
    frags.add Frag(text: text)
  # tokenisation de chaque fragment.
  var prevSpecial = true
  for f in frags:
    if f.isTok:
      result.add f.tok
      prevSpecial = true
      continue
    if f.text.len == 0: continue
    case t.kind
    of tkBpe:
      for w in preTokenize(f.text, t.pre):
        t.bpeWord(byteEncode(w), result)
    of tkSpm:
      var s = f.text
      if t.addSpacePrefix and prevSpecial: s = " " & s
      s = s.replace(" ", "\xE2\x96\x81")
      t.spmText(s, result)
    prevSpecial = false

proc tokenToPiece*(t: Tokenizer; id: int; renderSpecial = false): string =
  # Texte (octets UTF-8) correspondant à un token.
  if id < 0 or id >= t.tokens.len: return ""
  let s = t.tokens[id]
  case t.types[id]
  of ttControl, ttUnknown:
    return (if renderSpecial: s else: "")
  of ttUserDefined:
    return s
  of ttByte:
    if t.kind == tkSpm and s.len == 6 and s.startsWith("<0x"):
      return $char(parseHexInt(s[3..4]))
  else: discard
  case t.kind
  of tkBpe: byteDecode(s)
  of tkSpm: s.replace("\xE2\x96\x81", " ")

proc decode*(t: Tokenizer; ids: openArray[int]; renderSpecial = false): string =
  # Décode une suite de tokens en texte.
  for k, id in ids:
    var p = t.tokenToPiece(id, renderSpecial)
    if k == 0 and t.kind == tkSpm and t.addSpacePrefix and p.len > 0 and p[0] == ' ':
      p = p[1 .. ^1]
    result.add p

proc newBpeTokenizer*(tokens: seq[string]; merges: seq[string]; types: seq[TokenType];
                      pre = ptLlama3): Tokenizer =
  # Construit un tokeniseur BPE à partir d'un vocabulaire (forme " octets GPT-2 ").
  result = Tokenizer(kind: tkBpe, pre: pre, tokens: tokens, merges: merges,
                     types: types, bosId: -1, eosId: -1, unkId: -1, padId: -1)
  result.rebuild()
