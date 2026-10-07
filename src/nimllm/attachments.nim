# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/attachments — pièces jointes transmises au modèle.
#
# Un LLM textuel (comme llama3.2 1B/3B) ne « voit » que du texte : chaque
# pièce jointe est donc convertie en une description textuelle insérée dans
# le message :
#
# * texte, code, CSV, JSON, Markdown, HTML... : contenu intégral (tronqué) ;
# * PDF : extraction du texte des flux (FlateDecode géré) ;
# * images PNG/BMP/PPM : dimensions, couleurs dominantes et aperçu ASCII ;
#   JPEG/GIF/WebP : dimensions ;
# * WAV : format et durée ;
# * autres binaires : taille et aperçu hexadécimal.

import std/[os, strutils, tables, algorithm, math, unicode, json]
import inflate, image, audio

type
  AttachmentKind* = enum
    akText, akImage, akPdf, akAudio, akBinary

  Attachment* = object
    name*: string
    kind*: AttachmentKind
    mime*: string
    text*: string                # représentation textuelle envoyée au modèle.
    data*: string                # octets bruts.
    width*, height*: int         # pour les images.
    image*: Image                # pixels décodés (PNG/BMP/PPM), sinon nil.

const textExts = [".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml",
  ".html", ".htm", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log", ".nim",
  ".nims", ".nimble", ".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".js", ".ts", ".rs",
  ".go", ".java", ".kt", ".swift", ".rb", ".php", ".sh", ".bash", ".sql", ".css",
  ".tex", ".rst", ".srt", ".vtt", ".svg", ".lua", ".r", ".m", ".pl", ".bat", ".ps1"]

#
# PNG
#

proc be32(s: string; o: int): int =
  (ord(s[o]) shl 24) or (ord(s[o+1]) shl 16) or (ord(s[o+2]) shl 8) or ord(s[o+3])

proc decodePng*(data: string): Image =
  # Décode un PNG non entrelacé (8 ou 16 bits ; gris, RVB, palette, alpha).
  if data.len < 8 or data[0..7] != "\x89PNG\r\n\x1a\n":
    raise newException(IOError, "PNG invalide")
  var w, h, depth, ctype, interlace = 0
  var idat = ""
  var palette: seq[(uint8, uint8, uint8)]
  var i = 8
  while i + 8 <= data.len:
    let ln = be32(data, i)
    let typ = data[i+4 ..< i+8]
    let body = data[i+8 ..< min(data.len, i+8+ln)]
    case typ
    of "IHDR":
      w = be32(body, 0); h = be32(body, 4)
      depth = ord(body[8]); ctype = ord(body[9]); interlace = ord(body[12])
    of "PLTE":
      for k in 0 ..< body.len div 3:
        palette.add (uint8(body[3*k]), uint8(body[3*k+1]), uint8(body[3*k+2]))
    of "IDAT": idat.add body
    of "IEND": break
    else: discard
    i += 12 + ln
  if interlace != 0: raise newException(IOError, "PNG entrelacé non pris en charge")
  if depth notin [8, 16] and not (ctype == 3 and depth in [1, 2, 4, 8]) and
     not (ctype == 0 and depth in [1, 2, 4]):
    raise newException(IOError, "profondeur PNG non prise en charge")
  let raw = zlibDecompress(idat)
  let channels = case ctype
    of 0: 1
    of 2: 3
    of 3: 1
    of 4: 2
    of 6: 4
    else: raise newException(IOError, "type de couleur PNG inconnu")
  let bitsPP = channels * depth
  let bpp = max(1, bitsPP div 8)
  let stride = (w * bitsPP + 7) div 8
  result = Image(width: w, height: h, pixels: newSeq[uint8](w*h*3))
  var prev = newString(stride)
  var cur = newString(stride)
  var p = 0
  for y in 0 ..< h:
    if p + 1 + stride > raw.len: break
    let f = ord(raw[p]); inc p
    for x in 0 ..< stride:
      let a = if x >= bpp: ord(cur[x - bpp]) else: 0
      let b = ord(prev[x])
      let c = if x >= bpp: ord(prev[x - bpp]) else: 0
      var v = ord(raw[p + x])
      case f
      of 1: v += a
      of 2: v += b
      of 3: v += (a + b) div 2
      of 4:
        let pp = a + b - c
        let pa = abs(pp - a)
        let pb = abs(pp - b)
        let pc = abs(pp - c)
        v += (if pa <= pb and pa <= pc: a elif pb <= pc: b else: c)
      else: discard
      cur[x] = char(v and 0xFF)
    p += stride
    for x in 0 ..< w:
      var r, g, bl: uint8
      var alpha = 255
      if depth < 8:
        let bitpos = x * depth
        let byte = ord(cur[bitpos div 8])
        let shift = 8 - depth - (bitpos mod 8)
        let v = (byte shr shift) and ((1 shl depth) - 1)
        if ctype == 3 and v < palette.len: (r, g, bl) = palette[v]
        else:
          let gv = uint8(v * 255 div ((1 shl depth) - 1))
          r = gv; g = gv; bl = gv
      else:
        let step = depth div 8
        template ch(k: int): uint8 = uint8(cur[(x*channels + k)*step])
        case ctype
        of 0: r = ch(0); g = r; bl = r
        of 2: r = ch(0); g = ch(1); bl = ch(2)
        of 3:
          let v = ch(0).int
          if v < palette.len: (r, g, bl) = palette[v]
        of 4: r = ch(0); g = r; bl = r; alpha = ch(1).int
        of 6: r = ch(0); g = ch(1); bl = ch(2); alpha = ch(3).int
        else: discard
      # composition sur fond blanc.
      let o = (y*w + x)*3
      result.pixels[o] = uint8((r.int*alpha + 255*(255-alpha)) div 255)
      result.pixels[o+1] = uint8((g.int*alpha + 255*(255-alpha)) div 255)
      result.pixels[o+2] = uint8((bl.int*alpha + 255*(255-alpha)) div 255)
    swap(prev, cur)

proc imageDims(data: string): (string, int, int) =
  # Détecte le format et les dimensions sans décoder l'image.
  if data.len >= 24 and data.startsWith("\x89PNG"):
    return ("png", be32(data, 16), be32(data, 20))
  if data.len >= 10 and (data.startsWith("GIF87a") or data.startsWith("GIF89a")):
    return ("gif", ord(data[6]) or (ord(data[7]) shl 8), ord(data[8]) or (ord(data[9]) shl 8))
  if data.len >= 26 and data.startsWith("BM"):
    return ("bmp", int(cast[ptr int32](unsafeAddr data[18])[]), abs(int(cast[ptr int32](unsafeAddr data[22])[])))
  if data.len >= 4 and data.startsWith("\xFF\xD8"):
    var i = 2
    while i + 9 < data.len:
      if data[i] != '\xFF': inc i; continue
      let m = ord(data[i+1])
      if m in [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF]:
        let h = (ord(data[i+5]) shl 8) or ord(data[i+6])
        let w = (ord(data[i+7]) shl 8) or ord(data[i+8])
        return ("jpeg", w, h)
      let ln = (ord(data[i+2]) shl 8) or ord(data[i+3])
      i += 2 + ln
    return ("jpeg", 0, 0)
  if data.len >= 30 and data.startsWith("RIFF") and data[8..11] == "WEBP":
    if data[12..15] == "VP8X":
      let w = 1 + (ord(data[24]) or (ord(data[25]) shl 8) or (ord(data[26]) shl 16))
      let h = 1 + (ord(data[27]) or (ord(data[28]) shl 8) or (ord(data[29]) shl 16))
      return ("webp", w, h)
    if data[12..15] == "VP8 " and data.len >= 30:
      return ("webp", (ord(data[26]) or (ord(data[27]) shl 8)) and 0x3FFF,
              (ord(data[28]) or (ord(data[29]) shl 8)) and 0x3FFF)
    return ("webp", 0, 0)
  if data.len >= 2 and data.startsWith("P6"):
    return ("ppm", 0, 0)
  ("", 0, 0)

proc colorName(r, g, b: int): string =
  # Nom français approximatif d'une couleur.
  let mx = max(r, max(g, b))
  let mn = min(r, min(g, b))
  let l = (mx + mn) / 2
  if mx - mn < 25:
    if l > 220: return "blanc"
    if l > 160: return "gris clair"
    if l > 80: return "gris"
    if l > 35: return "gris foncé"
    return "noir"
  var hue = 0.0
  let d = (mx - mn).float
  if mx == r: hue = 60 * (((g - b).float / d) mod 6)
  elif mx == g: hue = 60 * ((b - r).float / d + 2)
  else: hue = 60 * ((r - g).float / d + 4)
  if hue < 0: hue += 360
  var name =
    if hue < 15 or hue >= 345: "rouge"
    elif hue < 40: "orange"
    elif hue < 70: "jaune"
    elif hue < 160: "vert"
    elif hue < 190: "cyan"
    elif hue < 255: "bleu"
    elif hue < 290: "violet"
    else: "rose"
  if name == "orange" and l < 90: name = "brun"
  if l < 70: name &= " foncé"
  elif l > 190: name &= " clair"
  name

proc describeImage*(img: Image; asciiWidth = 48): string =
  # Description textuelle : couleurs dominantes, luminosité, aperçu ASCII.
  let w = img.width
  let h = img.height
  var hist = initCountTable[(int, int, int)]()
  var sums = initTable[(int, int, int), (int, int, int)]()
  var lum = 0.0
  let step = max(1, int(sqrt((w*h).float / 40000)))
  var n = 0
  for y in countup(0, h-1, step):
    for x in countup(0, w-1, step):
      let (r, g, b) = img.getPixel(x, y)
      let key = (r.int div 51, g.int div 51, b.int div 51)
      hist.inc key
      let o = sums.getOrDefault(key)
      sums[key] = (o[0] + r.int, o[1] + g.int, o[2] + b.int)
      lum += 0.299*r.float + 0.587*g.float + 0.114*b.float
      inc n
  hist.sort()
  result.add "Luminosité moyenne : " & $int(lum / max(1, n).float / 2.55) & " %\n"
  result.add "Couleurs dominantes : "
  var k = 0
  var names: seq[string]
  for key, cnt in hist:
    if k >= 5: break
    let sm = sums[key]
    let (r, g, b) = (sm[0] div cnt, sm[1] div cnt, sm[2] div cnt)
    let nm = colorName(r, g, b)
    if cnt * 100 div max(1, n) == 0: break                        # couleurs marginales (< 1 %).
    if nm notin names:
      names.add nm
      result.add (if names.len > 1: ", " else: "") & nm & " (" & $(cnt * 100 div max(1, n)) & " %)"
    inc k
  result.add "\n"
  # aperçu ASCII (caractères du plus clair au plus sombre).
  const ramp = " .:-=+*#%@"
  let cols = min(asciiWidth, w)
  let rows = max(1, int(cols.float * h.float / w.float / 2.0))
  result.add "Aperçu (" & $cols & "×" & $rows & ", clair = espace, sombre = @) :\n"
  for ry in 0 ..< rows:
    var line = ""
    for rx in 0 ..< cols:
      let x0 = rx * w div cols
      let y0 = ry * h div rows
      let (r, g, b) = img.getPixel(min(w-1, x0), min(h-1, y0))
      let L = (0.299*r.float + 0.587*g.float + 0.114*b.float) / 255
      line.add ramp[min(ramp.high, int((1 - L) * ramp.len.float))]
    result.add line.strip(leading = false) & "\n"

#
# PDF (extraction de texte simple)
#

proc pdfUnescape(s: string): string =
  var i = 0
  while i < s.len:
    if s[i] == '\\' and i + 1 < s.len:
      let c = s[i+1]
      case c
      of 'n': result.add '\n'; i += 2
      of 'r': i += 2
      of 't': result.add '\t'; i += 2
      of '(', ')', '\\': result.add c; i += 2
      of '0'..'7':
        var j = i + 1
        var v = 0
        while j < s.len and j < i + 4 and s[j] in {'0'..'7'}:
          v = v*8 + ord(s[j]) - ord('0'); inc j
        # WinAnsi/Latin-1 -> UTF-8
        result.add $Rune(v)
        i = j
      else: i += 2
    else:
      let o = ord(s[i])
      if o >= 128: result.add $Rune(o) else: result.add s[i]
      inc i

proc extractPdfStrings(content: string): string =
  # Récupère le texte des opérateurs Tj/TJ/'/" d'un flux de contenu.
  var i = 0
  var line = ""
  while i < content.len:
    let c = content[i]
    if c == '(':
      var depth = 1
      var j = i + 1
      var s = ""
      while j < content.len and depth > 0:
        if content[j] == '\\' and j + 1 < content.len:
          s.add content[j]; s.add content[j+1]; j += 2; continue
        if content[j] == '(': inc depth
        elif content[j] == ')':
          dec depth
          if depth == 0: break
        s.add content[j]; inc j
      line.add pdfUnescape(s)
      i = j + 1
    elif c == ']' or c == '[':
      inc i
    elif content.continuesWith("ET", i) or content.continuesWith("T*", i) or
         content.continuesWith("Td", i) or content.continuesWith("TD", i):
      if line.len > 0 and not line.endsWith(" "): line.add " "
      if content.continuesWith("ET", i) or content.continuesWith("T*", i) or content.continuesWith("TD", i):
        result.add line.strip & "\n"; line = ""
      i += 2
    elif c == '-' or c.isDigit:
      # kerning important dans TJ (ex. -250) = espace.
      var j = i
      if content[j] == '-': inc j
      var v = 0
      while j < content.len and content[j].isDigit: v = v*10 + ord(content[j]) - ord('0'); inc j
      if content[i] == '-' and v >= 200 and line.len > 0 and not line.endsWith(" "): line.add " "
      i = max(j, i + 1)
    else: inc i
  if line.len > 0: result.add line.strip & "\n"

proc pdfText*(data: string): string =
  # Extrait (approximativement) le texte d'un PDF. Les PDF scannés (images).
  # ou utilisant des polices à encodage personnalisé ne sont pas lisibles.
  var i = 0
  while true:
    let s = data.find("stream", i)
    if s < 0: break
    var start = s + 6
    if start < data.len and data[start] == '\r': inc start
    if start < data.len and data[start] == '\n': inc start
    let e = data.find("endstream", start)
    if e < 0: break
    let dictStart = data.rfind("<<", last = s)
    let dict = if dictStart >= 0 and s - dictStart < 2000: data[dictStart ..< s] else: ""
    var content = data[start ..< e]
    var ok = true
    if "FlateDecode" in dict:
      try: content = zlibDecompress(content)
      except CatchableError: ok = false
    elif "Filter" in dict: ok = false
    if ok and ("Image" notin dict) and ("BT" in content):
      result.add extractPdfStrings(content)
    i = e + 9

#
# Construction des pièces jointes
#

proc looksLikeText(data: string): bool =
  if data.len == 0: return true
  let n = min(data.len, 4096)
  var bad = 0
  for i in 0 ..< n:
    let c = ord(data[i])
    if c == 0: return false
    if c < 9 or (c > 13 and c < 32): inc bad
  if bad * 100 >= n * 2: return false
  let v = validateUtf8(data[0 ..< n])
  v == -1 or (n < data.len and v >= n - 3)

proc attachText*(name, content: string): Attachment =
  # Pièce jointe textuelle construite en mémoire.
  Attachment(name: name, kind: akText, mime: "text/plain", text: content, data: content)

proc attachData*(name: string; data: string): Attachment =
  # Pièce jointe à partir d'octets en mémoire ; le type est détecté.
  result = Attachment(name: name, data: data)
  let ext = name.splitFile.ext.toLowerAscii
  let (fmt, w, h) = imageDims(data)
  if fmt.len > 0:
    result.kind = akImage
    result.mime = "image/" & fmt
    result.width = w; result.height = h
    var desc = "Image " & fmt.toUpperAscii
    try:
      case fmt
      of "png": result.image = decodePng(data)
      of "bmp":
        let tmp = getTempDir() / "nimllm_att.bmp"
        writeFile(tmp, data); result.image = readBmp(tmp); removeFile(tmp)
      of "ppm":
        let tmp = getTempDir() / "nimllm_att.ppm"
        writeFile(tmp, data); result.image = readPpm(tmp); removeFile(tmp)
      else: discard
    except CatchableError: discard
    if result.image != nil:
      result.width = result.image.width; result.height = result.image.height
    desc.add " de " & $result.width & "×" & $result.height & " pixels.\n"
    if result.image != nil:
      desc.add describeImage(result.image)
    else:
      desc.add "(contenu visuel non décodé : seules les dimensions sont connues)\n"
    result.text = desc
  elif data.startsWith("%PDF"):
    result.kind = akPdf
    result.mime = "application/pdf"
    let t = pdfText(data)
    result.text = if t.strip.len > 0: t else: "(PDF sans texte extractible)"
  elif data.startsWith("RIFF") and data.len > 12 and data[8..11] == "WAVE":
    result.kind = akAudio
    result.mime = "audio/wav"
    let tmp = getTempDir() / "nimllm_att.wav"
    writeFile(tmp, data)
    try:
      let info = readWavInfo(tmp)
      result.text = "Fichier audio WAV : " & $info.sampleRate & " Hz, " & $info.channels &
        " canal(aux), " & $info.bitsPerSample & " bits, durée " &
        formatFloat(info.durationSec, ffDecimal, 2) & " s. (Le contenu sonore n'est pas transcrit.)"
    except CatchableError:
      result.text = "Fichier audio WAV illisible."
    removeFile(tmp)
  elif ext in textExts or looksLikeText(data):
    result.kind = akText
    result.mime = "text/plain"
    result.text = data
  else:
    result.kind = akBinary
    result.mime = "application/octet-stream"
    var hex = ""
    for i in 0 ..< min(64, data.len): hex.add toHex(ord(data[i]), 2) & " "
    result.text = "Fichier binaire de " & $data.len & " octets. Premiers octets : " & hex

proc attach*(path: string): Attachment =
  # Charge un fichier joint depuis le disque.
  if not fileExists(path):
    raise newException(IOError, "pièce jointe introuvable : " & path)
  attachData(path.extractFilename, readFile(path))

proc toPrompt*(a: Attachment; maxChars = 12000): string =
  # Texte inséré dans le message envoyé au modèle.
  var body = a.text
  var note = ""
  if body.runeLen > maxChars:
    body = body.runeSubStr(0, maxChars)
    note = "\n[... contenu tronqué : " & $a.text.runeLen & " caractères au total]"
  let lang = case a.name.splitFile.ext.toLowerAscii
    of ".nim": "nim"
    of ".py": "python"
    of ".json": "json"
    of ".csv": "csv"
    of ".md": "markdown"
    of ".js": "javascript"
    of ".c", ".h": "c"
    else: ""
  case a.kind
  of akText:
    "### Pièce jointe : " & a.name & "\n```" & lang & "\n" & body & "\n```" & note
  of akPdf:
    "### Pièce jointe (PDF, texte extrait) : " & a.name & "\n" & body & note
  else:
    "### Pièce jointe : " & a.name & " (" & a.mime & ")\n" & body & note
