# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/audio — production de fichiers audio WAV sans dépendance :
#
# * `writeWav` / `readWavInfo` : fichiers PCM 16 bits ;
# * `speak` : synthèse vocale **rudimentaire** par formants (français ou
#   anglais), à voix robotique mais intelligible pour des phrases simples ;
# * `renderMelody` : mélodie à partir d'une notation texte (« C4 D4 E4:2 R »).
#
# La qualité n'égale pas un vrai moteur TTS neuronal : l'objectif est de
# fournir une sortie audio autonome, sans élément externe.

import std/[math, strutils, unicode, streams, random, tables]

type
  Audio* = object
    sampleRate*: int
    samples*: seq[float32]       # mono, valeurs -1..1

  WavInfo* = object
    sampleRate*, channels*, bitsPerSample*: int
    durationSec*: float

proc newAudio*(sampleRate = 16000): Audio = Audio(sampleRate: sampleRate)

proc duration*(a: Audio): float = a.samples.len.float / a.sampleRate.float

proc normalize*(a: var Audio; peak = 0.9) =
  var m = 0'f32
  for s in a.samples: m = max(m, abs(s))
  if m > 0:
    let k = peak.float32 / m
    for s in a.samples.mitems: s *= k

proc writeWav*(a: Audio; path: string) =
  # Écrit un fichier WAV PCM 16 bits mono.
  var s = newFileStream(path, fmWrite)
  if s == nil: raise newException(IOError, "impossible d'écrire " & path)
  defer: s.close()
  let n = a.samples.len
  s.write "RIFF"; s.write uint32(36 + n*2); s.write "WAVE"
  s.write "fmt "; s.write 16'u32; s.write 1'u16; s.write 1'u16
  s.write uint32(a.sampleRate); s.write uint32(a.sampleRate*2); s.write 2'u16; s.write 16'u16
  s.write "data"; s.write uint32(n*2)
  for v in a.samples:
    s.write int16(clamp(v, -1'f32, 1'f32) * 32767)

proc readWavInfo*(path: string): WavInfo =
  # Lit l'en-tête d'un fichier WAV (format, durée).
  let d = readFile(path)
  if d.len < 44 or d[0..3] != "RIFF" or d[8..11] != "WAVE":
    raise newException(IOError, "fichier WAV invalide : " & path)
  var i = 12
  var dataLen = 0
  while i + 8 <= d.len:
    let id = d[i ..< i+4]
    let sz = int(cast[ptr uint32](unsafeAddr d[i+4])[])
    if id == "fmt ":
      result.channels = int(cast[ptr uint16](unsafeAddr d[i+10])[])
      result.sampleRate = int(cast[ptr uint32](unsafeAddr d[i+12])[])
      result.bitsPerSample = int(cast[ptr uint16](unsafeAddr d[i+22])[])
    elif id == "data":
      dataLen = sz
    i += 8 + sz + (sz and 1)
  let bps = max(1, result.channels * result.bitsPerSample div 8)
  if result.sampleRate > 0:
    result.durationSec = dataLen.float / bps.float / result.sampleRate.float

proc readWav*(path: string): Audio =
  # Lit un WAV PCM 16 bits (mono ou stéréo -> mono).
  let d = readFile(path)
  let info = readWavInfo(path)
  result.sampleRate = info.sampleRate
  var i = 12
  while i + 8 <= d.len:
    let id = d[i ..< i+4]
    let sz = int(cast[ptr uint32](unsafeAddr d[i+4])[])
    if id == "data" and info.bitsPerSample == 16:
      let n = min(sz, d.len - i - 8) div (2 * info.channels)
      for k in 0 ..< n:
        var acc = 0.0
        for ch in 0 ..< info.channels:
          acc += float(cast[ptr int16](unsafeAddr d[i + 8 + (k*info.channels + ch)*2])[]) / 32768
        result.samples.add float32(acc / info.channels.float)
    i += 8 + sz + (sz and 1)

proc concat*(a: var Audio; b: Audio) = a.samples.add b.samples

proc silence*(a: var Audio; seconds: float) =
  for _ in 0 ..< int(seconds * a.sampleRate.float): a.samples.add 0

#
# Synthèse par formants
#

type
  PhKind = enum pkVowel, pkNasalVowel, pkNasal, pkLiquid, pkGlide,
                pkFricative, pkPlosive, pkPause
  Phone = object
    sym: string
    kind: PhKind
    f1, f2, f3: float
    voiced: bool
    noiseF, noiseBw: float       # bruit de friction / explosion.
    dur: float                   # secondes.
    amp: float

proc ph(sym: string; kind: PhKind; f1, f2, f3: float; voiced = true;
        noiseF = 0.0; noiseBw = 0.0; dur = 0.08; amp = 1.0): Phone =
  Phone(sym: sym, kind: kind, f1: f1, f2: f2, f3: f3, voiced: voiced,
        noiseF: noiseF, noiseBw: noiseBw, dur: dur, amp: amp)

let phones = {
  "a": ph("a", pkVowel, 750, 1300, 2600, dur = 0.11),
  "e": ph("e", pkVowel, 360, 2150, 2850, dur = 0.10),
  "E": ph("E", pkVowel, 530, 1800, 2550, dur = 0.10),
  "i": ph("i", pkVowel, 280, 2250, 3000, dur = 0.09),
  "o": ph("o", pkVowel, 380, 780, 2450, dur = 0.10),
  "O": ph("O", pkVowel, 520, 950, 2450, dur = 0.10),
  "u": ph("u", pkVowel, 300, 760, 2250, dur = 0.10),
  "y": ph("y", pkVowel, 280, 1800, 2250, dur = 0.09),
  "2": ph("2", pkVowel, 380, 1500, 2350, dur = 0.10),
  "@": ph("@", pkVowel, 480, 1450, 2450, dur = 0.07, amp = 0.8),
  "A~": ph("A~", pkNasalVowel, 650, 1050, 2550, dur = 0.13),
  "O~": ph("O~", pkNasalVowel, 430, 800, 2400, dur = 0.13),
  "E~": ph("E~", pkNasalVowel, 560, 1550, 2550, dur = 0.13),
  "m": ph("m", pkNasal, 260, 1000, 2300, dur = 0.07, amp = 0.55),
  "n": ph("n", pkNasal, 260, 1500, 2500, dur = 0.07, amp = 0.55),
  "J": ph("J", pkNasal, 260, 2000, 2800, dur = 0.08, amp = 0.55),
  "l": ph("l", pkLiquid, 360, 1300, 2600, dur = 0.06, amp = 0.75),
  "R": ph("R", pkLiquid, 500, 1250, 2300, noiseF = 1200, noiseBw = 1200, dur = 0.06, amp = 0.6),
  "w": ph("w", pkGlide, 300, 700, 2200, dur = 0.05, amp = 0.8),
  "j": ph("j", pkGlide, 270, 2250, 3000, dur = 0.05, amp = 0.8),
  "f": ph("f", pkFricative, 300, 1500, 2500, false, 5000, 4000, 0.09, 0.35),
  "v": ph("v", pkFricative, 300, 1500, 2500, true, 4500, 4000, 0.07, 0.4),
  "s": ph("s", pkFricative, 300, 1700, 2600, false, 6500, 2500, 0.10, 0.6),
  "z": ph("z", pkFricative, 300, 1700, 2600, true, 6000, 2500, 0.08, 0.5),
  "S": ph("S", pkFricative, 300, 1700, 2500, false, 3200, 1500, 0.10, 0.6),
  "Z": ph("Z", pkFricative, 300, 1700, 2500, true, 3000, 1500, 0.08, 0.5),
  "p": ph("p", pkPlosive, 300, 900, 2300, false, 900, 1500, 0.08, 0.6),
  "b": ph("b", pkPlosive, 300, 900, 2300, true, 800, 1500, 0.07, 0.5),
  "t": ph("t", pkPlosive, 300, 1700, 2600, false, 4000, 2500, 0.08, 0.6),
  "d": ph("d", pkPlosive, 300, 1700, 2600, true, 3500, 2500, 0.07, 0.5),
  "k": ph("k", pkPlosive, 300, 1900, 2500, false, 2200, 1500, 0.08, 0.6),
  "g": ph("g", pkPlosive, 300, 1900, 2500, true, 2000, 1500, 0.07, 0.5),
  "_": ph("_", pkPause, 500, 1500, 2500, false, dur = 0.25, amp = 0),
  ",": ph(",", pkPause, 500, 1500, 2500, false, dur = 0.18, amp = 0),
  "|": ph("|", pkPause, 500, 1500, 2500, false, dur = 0.015, amp = 0),
}.toTable

# nombres en toutes lettres

proc numberToFrench*(n: int): string =
  ## Écrit un entier (0..999 999 999) en toutes lettres.
  const units = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
                 "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
                 "dix-sept", "dix-huit", "dix-neuf"]
  const tens = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante",
                "soixante", "quatre-vingt", "quatre-vingt"]
  proc below100(n: int): string =
    if n < 20: return units[n]
    let t = n div 10
    var u = n mod 10
    if t == 7 or t == 9: u += 10
    result = tens[t]
    if u == 0: return (if t == 8: result & "s" else: result)
    if u == 1 and t != 8 and t != 9: result.add " et un"
    elif u == 11 and t == 7: result.add " et onze"
    else: result.add "-" & units[u]
  proc below1000(n: int): string =
    let c = n div 100
    let r = n mod 100
    if c == 0: return below100(r)
    result = if c == 1: "cent" else: units[c] & " cent"
    if r == 0 and c > 1: result.add "s"
    elif r > 0: result.add " " & below100(r)
  if n < 0: return "moins " & numberToFrench(-n)
  if n < 1000: return below1000(n)
  if n < 1_000_000:
    let k = n div 1000
    let r = n mod 1000
    result = if k == 1: "mille" else: below1000(k) & " mille"
    if r > 0: result.add " " & below1000(r)
    return
  let m = n div 1_000_000
  let r = n mod 1_000_000
  result = below1000(m) & (if m == 1: " million" else: " millions")
  if r > 0: result.add " " & numberToFrench(r)

proc numberToEnglish*(n: int): string =
  const units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                 "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                 "sixteen", "seventeen", "eighteen", "nineteen"]
  const tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
  if n < 0: return "minus " & numberToEnglish(-n)
  if n < 20: return units[n]
  if n < 100: return tens[n div 10] & (if n mod 10 > 0: " " & units[n mod 10] else: "")
  if n < 1000:
    return units[n div 100] & " hundred" & (if n mod 100 > 0: " " & numberToEnglish(n mod 100) else: "")
  if n < 1_000_000:
    return numberToEnglish(n div 1000) & " thousand" &
      (if n mod 1000 > 0: " " & numberToEnglish(n mod 1000) else: "")
  numberToEnglish(n div 1_000_000) & " million" &
    (if n mod 1_000_000 > 0: " " & numberToEnglish(n mod 1_000_000) else: "")

proc spellNumbers(text: string; lang: string): string =
  var i = 0
  while i < text.len:
    if text[i].isDigit:
      var j = i
      while j < text.len and text[j].isDigit: inc j
      let s = text[i ..< j]
      if s.len <= 9:
        let n = parseInt(s)
        result.add " " & (if lang == "fr": numberToFrench(n) else: numberToEnglish(n)) & " "
      else:
        for ch in s:
          let n = ord(ch) - ord('0')
          result.add " " & (if lang == "fr": numberToFrench(n) else: numberToEnglish(n))
      i = j
    else:
      result.add text[i]; inc i

# graphèmes -> phonèmes

proc stripAccentsKeep(r: Rune): string =
  let s = $r
  case s
  of "à", "â", "ä": "a"
  of "î", "ï": "i"
  of "ô", "ö": "o"
  of "ù", "û", "ü": "u"
  of "ç": "ç"
  of "é": "é"
  of "è", "ê", "ë": "è"
  of "œ": "oe"
  of "æ": "ae"
  of "ÿ": "y"
  else: s

proc isV(c: char): bool = c in {'a', 'e', 'i', 'o', 'u', 'y'}

proc frenchWord(w: string): seq[string] =
  # Règles simplifiées de prononciation du français.
  var s = ""
  for r in w.toLower.runes: s.add stripAccentsKeep(r)
  # remplace é/è par des marqueurs ASCII.
  s = s.replace("é", "1").replace("è", "2").replace("ç", "3")
  let n = s.len
  var i = 0
  template at(k: int): char = (if k >= 0 and k < n: s[k] else: '\0')
  template rest(k: int): string = (if k < n: s[k .. ^1] else: "")
  template isEnd(k: int): bool = k >= n
  template nasalNext(k: int): bool =
    # voyelle + n/m suivie d'une consonne ou fin de mot (pas de nn/mm).
    (at(k) in {'n', 'm'}) and not isV(at(k+1)) and at(k+1) notin {'n', 'm', '1', '2'}
  while i < n:
    let c = s[i]
    # e muet / consonnes finales muettes.
    if i == n - 1 and c == 'e' and n > 2:
      inc i; continue
    if i == n - 2 and c == 'e' and at(i+1) == 's' and n > 3:
      i += 2; continue
    if i == n - 3 and rest(i) == "ent" and n > 4:
      # « -ent » des verbes : muet (approximation).
      i += 3; continue
    if i == n - 1 and c in {'s', 't', 'd', 'x', 'z', 'p', 'g'} and n > 1:
      inc i; continue
    if i == n - 2 and rest(i) in ["gt", "ds", "ts", "ps", "ct"] and n > 3:
      i += 2; continue
    if i == n - 2 and rest(i) == "er" and n > 3:
      result.add "e"; i += 2; continue
    if i == n - 2 and rest(i) == "ez":
      result.add "e"; i += 2; continue
    if rest(i).startsWith("eau"): result.add "o"; i += 3; continue
    if rest(i).startsWith("oin") and nasalNext(i+2): result.add ["w", "E~"]; i += 3; continue
    if rest(i).startsWith("oi"): result.add ["w", "a"]; i += 2; continue
    if rest(i).startsWith("ou"): result.add "u"; i += 2; continue
    if rest(i).startsWith("au"): result.add "o"; i += 2; continue
    if (rest(i).startsWith("ain") or rest(i).startsWith("ein")) and nasalNext(i+2):
      result.add "E~"; i += 3; continue
    if rest(i).startsWith("ai") or rest(i).startsWith("ei"):
      result.add "E"; i += 2; continue
    if rest(i).startsWith("eu") or rest(i).startsWith("oeu"):
      result.add "2"; i += (if c == 'o': 3 else: 2); continue
    if rest(i).startsWith("ill") and i > 0:
      result.add ["i", "j"]; i += 3; continue
    if c in {'a', 'e'} and nasalNext(i+1):
      result.add "A~"; i += 2; continue
    if c == 'o' and nasalNext(i+1):
      result.add "O~"; i += 2; continue
    if c in {'i', 'y'} and nasalNext(i+1):
      result.add "E~"; i += 2; continue
    if c == 'u' and nasalNext(i+1):
      result.add "E~"; i += 2; continue
    if rest(i).startsWith("ch"): result.add "S"; i += 2; continue
    if rest(i).startsWith("ph"): result.add "f"; i += 2; continue
    if rest(i).startsWith("th"): result.add "t"; i += 2; continue
    if rest(i).startsWith("gn"): result.add "J"; i += 2; continue
    if rest(i).startsWith("qu"): result.add "k"; i += 2; continue
    if rest(i).startsWith("gu") and at(i+2) in {'e', 'i', '1', '2'}: result.add "g"; i += 2; continue
    case c
    of 'a': result.add "a"
    of '1': result.add "e"
    of '2': result.add "E"
    of '3': result.add "s"
    of 'e':
      # e devant deux consonnes ou consonne finale prononcée -> è.
      if at(i+1) in {'b', 'c', 'd', 'f', 'g', 'k', 'p', 't', 'v'} and at(i+2) in {'r', 'l'}:
        result.add "@"
      elif not isV(at(i+1)) and at(i+1) != '\0' and not isV(at(i+2)) and at(i+2) != '\0' and
         at(i+1) != at(i+2) or (at(i+1) == at(i+2) and at(i+1) in {'s', 't', 'l', 'r', 'n', 'm'}):
        result.add "E"
      elif at(i+1) in {'r', 'l', 'c', 'f'} and isEnd(i+2):
        result.add "E"
      elif at(i+1) == 't' and isEnd(i+2):
        result.add "e"; inc i
      else: result.add "@"
    of 'i', 'y':
      if isV(at(i+1)) and i > 0: result.add "j" else: result.add "i"
    of 'o': result.add (if not isV(at(i+1)) and at(i+1) != '\0' and not isEnd(i+2): "O" else: "o")
    of 'u': result.add "y"
    of 'b': result.add "b"
    of 'c':
      if at(i+1) in {'e', 'i', 'y', '1', '2'}: result.add "s" else: result.add "k"
    of 'd': result.add "d"
    of 'f': result.add "f"
    of 'g':
      if at(i+1) in {'e', 'i', 'y', '1', '2'}: result.add "Z" else: result.add "g"
    of 'h': discard
    of 'j': result.add "Z"
    of 'k', 'q': result.add "k"
    of 'l': result.add "l"
    of 'm': result.add "m"
    of 'n': result.add "n"
    of 'p': result.add "p"
    of 'r': result.add "R"
    of 's':
      if isV(at(i-1)) and isV(at(i+1)) or (at(i-1) in {'1', '2'} and isV(at(i+1))): result.add "z"
      else: result.add "s"
    of 't': result.add "t"
    of 'v': result.add "v"
    of 'w': result.add "w"
    of 'x': result.add ["k", "s"]
    of 'z': result.add "z"
    else: discard
    # consonnes doubles
    if c notin {'a', 'e', 'i', 'o', 'u', 'y', '1', '2'} and at(i+1) == c: inc i
    inc i

proc englishWord(w: string): seq[string] =
  # Règles (très) approximatives pour l'anglais.
  let s = w.toLowerAscii
  let n = s.len
  var i = 0
  template rest(k: int): string = (if k < n: s[k .. ^1] else: "")
  template at(k: int): char = (if k >= 0 and k < n: s[k] else: '\0')
  while i < n:
    let c = s[i]
    if i == n - 1 and c == 'e' and n > 2: inc i; continue
    if rest(i).startsWith("th"): result.add "z"; i += 2; continue
    if rest(i).startsWith("sh"): result.add "S"; i += 2; continue
    if rest(i).startsWith("ch"): result.add ["t", "S"]; i += 2; continue
    if rest(i).startsWith("ph"): result.add "f"; i += 2; continue
    if rest(i).startsWith("ng"): result.add "n"; i += 2; continue
    if rest(i).startsWith("ee") or rest(i).startsWith("ea"): result.add "i"; i += 2; continue
    if rest(i).startsWith("oo"): result.add "u"; i += 2; continue
    if rest(i).startsWith("ou") or rest(i).startsWith("ow"): result.add ["a", "u"]; i += 2; continue
    if rest(i).startsWith("ai") or rest(i).startsWith("ay"): result.add ["e", "i"]; i += 2; continue
    if rest(i).startsWith("oa"): result.add "o"; i += 2; continue
    if rest(i).startsWith("igh"): result.add ["a", "i"]; i += 3; continue
    case c
    of 'a': result.add (if at(i+2) == 'e' and not (at(i+1) in {'a','e','i','o','u'}): "e" else: "a")
    of 'e': result.add "E"
    of 'i': result.add (if at(i+2) == 'e' and i + 3 == n: "a" else: "i")
    of 'o': result.add (if at(i+2) == 'e' and i + 3 == n: "o" else: "O")
    of 'u': result.add "@"
    of 'y': result.add (if i == 0: "j" else: "i")
    of 'c': result.add (if at(i+1) in {'e', 'i', 'y'}: "s" else: "k")
    of 'g': result.add (if at(i+1) in {'e', 'i'}: "Z" else: "g")
    of 'j': result.add ["d", "Z"]
    of 'q': result.add "k"
    of 'r': result.add "R"
    of 'x': result.add ["k", "s"]
    of 'h': result.add "_h"
    of 'b', 'd', 'f', 'k', 'l', 'm', 'n', 'p', 's', 't', 'v', 'w', 'z': result.add $c
    else: discard
    if c notin {'a', 'e', 'i', 'o', 'u'} and at(i+1) == c: inc i
    inc i

proc textToPhonemes*(text: string; lang = "fr"): seq[string] =
  # Transcription (approximative) texte -> symboles phonétiques internes.
  let t = spellNumbers(text, lang)
  var word = ""
  proc flush(res: var seq[string]) =
    if word.len > 0:
      let p = if lang == "fr": frenchWord(word) else: englishWord(word)
      for x in p:
        if x != "_h": res.add x
      res.add "|"
      word = ""
  for r in t.runes:
    if r.isAlpha or $r == "'" and false:
      word.add $r
    else:
      flush(result)
      let s = $r
      if s in [".", "!", "?", ";", ":", "\n"]: result.add "_"
      elif s in [",", "(", ")", "-"]: result.add ","
  flush(result)

type Resonator = object
  a, b, c, y1, y2: float

proc setRes(r: var Resonator; f, bw, sr: float) =
  let T = 1.0 / sr
  r.c = -exp(-2*PI*bw*T)
  r.b = 2*exp(-PI*bw*T)*cos(2*PI*f*T)
  r.a = 1 - r.b - r.c

proc run(r: var Resonator; x: float): float {.inline.} =
  result = r.a*x + r.b*r.y1 + r.c*r.y2
  r.y2 = r.y1
  r.y1 = result

proc speak*(text: string; lang = "fr"; pitch = 115.0; speed = 1.0;
            sampleRate = 16000): Audio =
  # Synthèse vocale par formants. `pitch` : fréquence fondamentale (Hz),
  # `speed` : > 1 plus rapide. `lang` : "fr" ou "en".
  result = newAudio(sampleRate)
  let sr = sampleRate.float
  let phs = textToPhonemes(text, lang)
  var r1, r2, r3, r4, rn: Resonator
  var phase = 0.0
  var prev = phones["_"]
  var rng = initRand(1234)
  var total = 0.0
  for p in phs:
    if p in phones: total += phones[p].dur / speed
  var t = 0.0
  for idx, sym in phs:
    if sym notin phones: continue
    let cur = phones[sym]
    let n = int(cur.dur / speed * sr)
    # question : intonation montante en fin de phrase.
    var rising = false
    for k in idx+1 ..< phs.len:
      if phs[k] == "_": break
    if idx + 1 < phs.len and phs[idx+1] == "_" and text.strip.endsWith("?"): rising = true
    for k in 0 ..< n:
      let frac = k.float / max(1, n).float
      # interpolation des formants depuis le phonème précédent.
      let w = if prev.kind == pkPause: 1.0 else: min(1.0, frac / 0.35)
      let f1 = prev.f1 + (cur.f1 - prev.f1) * w
      let f2 = prev.f2 + (cur.f2 - prev.f2) * w
      let f3 = prev.f3 + (cur.f3 - prev.f3) * w
      if k mod 16 == 0:
        r1.setRes(f1, 80, sr); r2.setRes(f2, 100, sr); r3.setRes(f3, 140, sr)
        r4.setRes(3500, 250, sr)
        if cur.noiseF > 0: rn.setRes(min(cur.noiseF, sr*0.45), cur.noiseBw, sr)
      # hauteur : légère déclinaison + vibrato.
      var f0 = pitch * (1.08 - 0.16 * (t / max(0.01, total))) * (1 + 0.01*sin(2*PI*5*t))
      if rising: f0 *= 1.0 + 0.4*frac
      phase += f0 / sr
      if phase >= 1: phase -= 1
      # source glottique (impulsion de Rosenberg dérivée).
      var glott = 0.0
      if phase < 0.6: glott = 0.5*(1 - cos(PI*phase/0.6))
      elif phase < 0.75: glott = cos(PI*(phase - 0.6)/0.3)
      let noise = rng.rand(2.0) - 1.0
      var sample = 0.0
      # enveloppe.
      var env = 1.0
      let edge = 0.012 * sr
      if k.float < edge: env = k.float / edge
      if (n - k).float < edge: env = min(env, (n - k).float / edge)
      case cur.kind
      of pkPause:
        sample = 0
      of pkVowel, pkGlide, pkLiquid, pkNasalVowel, pkNasal:
        var src = glott - 0.5
        var v = r4.run(r3.run(r2.run(r1.run(src))))
        if cur.kind in {pkNasal, pkNasalVowel}: v *= (if cur.kind == pkNasal: 0.6 else: 0.85)
        if cur.noiseF > 0: v += 0.15 * rn.run(noise)
        sample = v * cur.amp * env
      of pkFricative:
        var v = rn.run(noise) * 0.9
        if cur.voiced: v = v*0.6 + 0.4*r2.run(r1.run(glott - 0.5))
        sample = v * cur.amp * env
      of pkPlosive:
        # occlusion (silence ou barre de voisement) puis explosion.
        if frac < 0.55:
          sample = if cur.voiced: 0.15*r1.run(glott - 0.5) else: 0.0
        elif frac < 0.8:
          sample = rn.run(noise) * cur.amp * 1.5
        else:
          let v = r3.run(r2.run(r1.run(if cur.voiced: glott - 0.5 else: noise*0.3)))
          sample = v * cur.amp * 0.6
      result.samples.add float32(sample)
      t += 1.0 / sr
    if cur.kind != pkPause or sym == "_": prev = cur
  # filtre passe-bas doux et normalisation.
  var lp = 0.0
  for s in result.samples.mitems:
    lp = lp * 0.35 + s.float * 0.65
    s = float32(lp)
  result.normalize(0.85)

#
# Mélodies
#

proc noteFrequency*(note: string): float =
  # « A4 » -> 440 Hz ; accepte les dièses (C#4) et bémols (Bb3).
  const names = {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}.toTable
  if note.len < 2: return 0
  let l = note[0].toUpperAscii
  if l notin names: return 0
  var semi = names[l]
  var i = 1
  if i < note.len and note[i] == '#': inc semi; inc i
  elif i < note.len and note[i] == 'b': dec semi; inc i
  var octave = 4
  try: octave = parseInt(note[i .. ^1]) except ValueError: discard
  440.0 * pow(2.0, (semi + 12*(octave - 4)).float / 12.0)

proc renderMelody*(notation: string; bpm = 120.0; sampleRate = 22050): Audio =
  # Notation : suite de « Note[:temps] » séparées par des espaces, R = silence.
  # Exemple : "C4 D4 E4 F4 G4:2 R E4:0.5 C4:1.5".
  result = newAudio(sampleRate)
  let beat = 60.0 / bpm
  for tok in strutils.splitWhitespace(notation):
    let parts = tok.split(':')
    var beats = 1.0
    if parts.len > 1:
      try: beats = parseFloat(parts[1]) except ValueError: discard
    let dur = beats * beat
    let n = int(dur * sampleRate.float)
    let f = if parts[0].toUpperAscii == "R": 0.0 else: noteFrequency(parts[0])
    for k in 0 ..< n:
      let t = k.float / sampleRate.float
      var env = 1.0
      if t < 0.01: env = t / 0.01
      env *= exp(-2.5 * t / max(0.2, dur))
      if dur - t < 0.02: env *= max(0.0, (dur - t) / 0.02)
      var v = 0.0
      if f > 0:
        v = sin(2*PI*f*t) + 0.4*sin(4*PI*f*t) + 0.15*sin(6*PI*f*t)
      result.samples.add float32(v * env * 0.4)
  result.normalize(0.8)
