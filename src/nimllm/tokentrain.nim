# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/tokentrain — entraînement d'un tokeniseur BPE niveau octet
# (même pré-découpage que Llama 3), compatible llama.cpp.

import std/[tables, sets, heapqueue, sequtils]
import tokenizer

const chatmlTemplate* = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  ## Gabarit Jinja ChatML enregistré dans le GGUF (utilisé par llama.cpp).

const defaultSpecials* = @["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]

iterator runeStrings(s: string): string =
  var i = 0
  while i < s.len:
    let c = ord(s[i])
    let n = if c < 0x80: 1 elif c shr 5 == 6: 2 elif c shr 4 == 14: 3 else: 4
    yield s[i ..< min(s.len, i + n)]
    i += n

type
  Word = object
    syms: seq[int]
    freq: int

proc trainBpe*(texts: openArray[string]; vocabSize = 2048;
               specials: seq[string] = defaultSpecials; minFreq = 2;
               pre = ptLlama3; verbose = false): Tokenizer =
  # Apprend un vocabulaire BPE de `vocabSize` tokens (256 octets + fusions
  # + tokens spéciaux) à partir des textes fournis.
  #
  # Les tokens spéciaux reçoivent le type CONTROL ; `<|bos|>`/`<|eos|>` (ou
  # les premiers spéciaux) deviennent les tokens de début/fin et
  # `<|im_end|>` marque la fin d'un tour de parole (gabarit ChatML).
  # comptage des mots pré-découpés.
  var wordFreq = initCountTable[string]()
  for t in texts:
    for w in preTokenize(t, pre):
      wordFreq.inc byteEncode(w)
  # symboles initiaux = 256 octets.
  var vocab: seq[string]
  var symId = initTable[string, int]()
  for b in 0 .. 255:
    symId[byteToUni[b]] = vocab.len
    vocab.add byteToUni[b]
  var words: seq[Word]
  for w, f in wordFreq:
    var wd = Word(freq: f)
    for r in w.runeStrings: wd.syms.add symId[r]
    words.add wd
  # comptage des paires + index des mots qui les contiennent.
  var pairCount = initTable[(int, int), int]()
  var where = initTable[(int, int), HashSet[int]]()
  for i, w in words:
    for k in 0 ..< w.syms.len - 1:
      let p = (w.syms[k], w.syms[k+1])
      pairCount.mgetOrPut(p, 0) += w.freq
      where.mgetOrPut(p, initHashSet[int]()).incl i
  var heap = initHeapQueue[(int, int, int)]()                     # (-compte, a, b)
  for p, c in pairCount: heap.push (-c, p[0], p[1])
  var merges: seq[string]
  let target = vocabSize - specials.len
  while vocab.len < target and heap.len > 0:
    let (negc, a, b) = heap.pop()
    let p = (a, b)
    let cur = pairCount.getOrDefault(p, 0)
    if cur != -negc:
      if cur > 0: heap.push (-cur, a, b)                          # compte périmé : on réinsère.
      continue
    if cur < minFreq: break
    let newId = vocab.len
    let newSym = vocab[a] & vocab[b]
    vocab.add newSym
    symId[newSym] = newId
    merges.add vocab[a] & " " & vocab[b]
    var touched = initHashSet[(int, int)]()
    for wi in where.getOrDefault(p, initHashSet[int]()).toSeq:
      var w = words[wi]
      # retire les anciennes paires du mot.
      for k in 0 ..< w.syms.len - 1:
        let q = (w.syms[k], w.syms[k+1])
        pairCount[q] = pairCount.getOrDefault(q) - w.freq
        touched.incl q
      # applique la fusion.
      var ns: seq[int]
      var k = 0
      while k < w.syms.len:
        if k + 1 < w.syms.len and w.syms[k] == a and w.syms[k+1] == b:
          ns.add newId; k += 2
        else:
          ns.add w.syms[k]; inc k
      w.syms = ns
      words[wi] = w
      for k2 in 0 ..< w.syms.len - 1:
        let q = (w.syms[k2], w.syms[k2+1])
        pairCount.mgetOrPut(q, 0) += w.freq
        where.mgetOrPut(q, initHashSet[int]()).incl wi
        touched.incl q
    pairCount.del p
    where.del p
    for q in touched:
      let c = pairCount.getOrDefault(q, 0)
      if c > 0: heap.push (-c, q[0], q[1])
    if verbose and merges.len mod 500 == 0:
      echo "  fusions : ", merges.len, " / ", target - 256
  # tokens spéciaux.
  var types = newSeq[TokenType](vocab.len)
  for x in types.mitems: x = ttNormal
  for s in specials:
    vocab.add s
    types.add ttControl
  let tk = newBpeTokenizer(vocab, merges, types, pre)
  proc idOf(names: openArray[string]): int =
    for n in names:
      let i = tk.tokenId(n)
      if i >= 0: return i
    -1
  result = tk
  result.bosId = idOf(["<|bos|>", "<s>", "<|begin_of_text|>"])
  result.eosId = idOf(["<|eos|>", "</s>", "<|end_of_text|>", "<|endoftext|>"])
  result.padId = idOf(["<|pad|>"])
  result.addBos = result.bosId >= 0
  if result.tokenId("<|im_start|>") >= 0: result.chatTemplate = chatmlTemplate
  result.rebuild()

proc byteLevelTokenizer*(specials: seq[string] = defaultSpecials): Tokenizer =
  # Tokeniseur minimal : un token par octet (aucun apprentissage nécessaire).
  trainBpe([""], 256 + specials.len, specials)
