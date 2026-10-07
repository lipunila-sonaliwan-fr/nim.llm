# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/image — images matricielles minimalistes et rendu SVG.
#
# * `Image` : tampon RVB 8 bits, écriture BMP (24 bits) et PPM ;
# * `Canvas` : primitives de dessin (rectangles, cercles, polygones, traits) ;
# * `renderSvg` : rastérise un sous-ensemble de SVG (rect, circle, ellipse,
#   line, polyline, polygon, path M/L/H/V/C/S/Q/T/A/Z, g, transform, styles).
#
# Permet de transformer en vraie image le code SVG produit par un LLM.

import std/[strutils, math, tables, streams, parseutils, algorithm]

type
  Color* = object
    r*, g*, b*, a*: float32      # composantes 0..1

  Image* = ref object
    width*, height*: int
    pixels*: seq[uint8]          # RVB, ligne par ligne, de haut en bas.

  Point* = tuple[x, y: float]

  Affine* = array[6, float]      # matrice [a b c d e f] (SVG).

  Canvas* = ref object
    img*: Image
    ss*: int                     # facteur de suréchantillonnage (anticrénelage).
    buf: seq[float32]            # RVB en flottants, taille (w*ss)*(h*ss)*3
    bw, bh: int

proc rgb*(r, g, b: int; a = 1.0): Color =
  Color(r: r.float32 / 255, g: g.float32 / 255, b: b.float32 / 255, a: a.float32)

const namedColors = {
  "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
  "green": (0, 128, 0), "lime": (0, 255, 0), "blue": (0, 0, 255),
  "yellow": (255, 255, 0), "cyan": (0, 255, 255), "aqua": (0, 255, 255),
  "magenta": (255, 0, 255), "fuchsia": (255, 0, 255), "gray": (128, 128, 128),
  "grey": (128, 128, 128), "silver": (192, 192, 192), "maroon": (128, 0, 0),
  "olive": (128, 128, 0), "purple": (128, 0, 128), "teal": (0, 128, 128),
  "navy": (0, 0, 128), "orange": (255, 165, 0), "pink": (255, 192, 203),
  "brown": (165, 42, 42), "gold": (255, 215, 0), "skyblue": (135, 206, 235),
  "lightblue": (173, 216, 230), "darkblue": (0, 0, 139), "darkgreen": (0, 100, 0),
  "lightgreen": (144, 238, 144), "forestgreen": (34, 139, 34), "lightgray": (211, 211, 211),
  "lightgrey": (211, 211, 211), "darkgray": (169, 169, 169), "darkgrey": (169, 169, 169),
  "beige": (245, 245, 220), "tan": (210, 180, 140), "violet": (238, 130, 238),
  "indigo": (75, 0, 130), "coral": (255, 127, 80), "salmon": (250, 128, 114),
  "crimson": (220, 20, 60), "khaki": (240, 230, 140), "turquoise": (64, 224, 208),
  "chocolate": (210, 105, 30), "sienna": (160, 82, 45), "orangered": (255, 69, 0),
  "steelblue": (70, 130, 180), "royalblue": (65, 105, 225), "midnightblue": (25, 25, 112),
  "seagreen": (46, 139, 87), "limegreen": (50, 205, 50), "darkred": (139, 0, 0),
  "lavender": (230, 230, 250), "ivory": (255, 255, 240), "wheat": (245, 222, 179),
  "darkorange": (255, 140, 0), "deepskyblue": (0, 191, 255), "dodgerblue": (30, 144, 255),
  "hotpink": (255, 105, 180), "plum": (221, 160, 221), "orchid": (218, 112, 214),
  "slategray": (112, 128, 144), "whitesmoke": (245, 245, 245), "snow": (255, 250, 250),
  "yellowgreen": (154, 205, 50), "peru": (205, 133, 63), "firebrick": (178, 34, 34),
  "goldenrod": (218, 165, 32), "lightyellow": (255, 255, 224), "mintcream": (245, 255, 250),
  "darkslategray": (47, 79, 79), "dimgray": (105, 105, 105), "sandybrown": (244, 164, 96)
}.toTable

proc parseColor*(s: string; current = rgb(0, 0, 0)): (bool, Color) =
  # Analyse une couleur CSS/SVG. Retourne (false, _) pour "none".
  let t = s.strip.toLowerAscii
  if t.len == 0 or t == "none" or t == "transparent": return (false, Color())
  if t == "currentcolor": return (true, current)
  if t[0] == '#':
    let h = t[1 .. ^1]
    try:
      if h.len == 3:
        return (true, rgb(parseHexInt($h[0] & h[0]), parseHexInt($h[1] & h[1]),
                          parseHexInt($h[2] & h[2])))
      if h.len >= 6:
        return (true, rgb(parseHexInt(h[0..1]), parseHexInt(h[2..3]), parseHexInt(h[4..5])))
    except ValueError: discard
    return (true, rgb(0, 0, 0))
  if t.startsWith("rgb"):
    let inner = t[t.find('(')+1 ..< max(t.find('(')+1, t.rfind(')'))]
    var vals: seq[float]
    for p in inner.split(','):
      var x = p.strip
      var pct = false
      if x.endsWith("%"): pct = true; x = x[0 .. ^2]
      try:
        var v = parseFloat(x)
        if pct: v = v * 2.55
        vals.add v
      except ValueError: vals.add 0
    while vals.len < 3: vals.add 0
    let a = if vals.len >= 4: vals[3] else: 1.0
    return (true, rgb(int(vals[0]), int(vals[1]), int(vals[2]), a))
  if t in namedColors:
    let (r, g, b) = namedColors[t]
    return (true, rgb(r, g, b))
  (true, rgb(0, 0, 0))

#
# Image
#

proc newImage*(w, h: int; bg = rgb(255, 255, 255)): Image =
  result = Image(width: w, height: h, pixels: newSeq[uint8](w*h*3))
  for i in 0 ..< w*h:
    result.pixels[3*i] = uint8(bg.r * 255)
    result.pixels[3*i+1] = uint8(bg.g * 255)
    result.pixels[3*i+2] = uint8(bg.b * 255)

proc getPixel*(img: Image; x, y: int): (uint8, uint8, uint8) =
  let o = (y*img.width + x)*3
  (img.pixels[o], img.pixels[o+1], img.pixels[o+2])

proc setPixel*(img: Image; x, y: int; c: Color) =
  if x < 0 or y < 0 or x >= img.width or y >= img.height: return
  let o = (y*img.width + x)*3
  let a = c.a
  img.pixels[o] = uint8(clamp(c.r*a*255 + img.pixels[o].float32*(1-a), 0, 255))
  img.pixels[o+1] = uint8(clamp(c.g*a*255 + img.pixels[o+1].float32*(1-a), 0, 255))
  img.pixels[o+2] = uint8(clamp(c.b*a*255 + img.pixels[o+2].float32*(1-a), 0, 255))

proc writeBmp*(img: Image; path: string) =
  # Enregistre l'image au format BMP 24 bits (lisible partout).
  let rowSize = (img.width * 3 + 3) div 4 * 4
  let dataSize = rowSize * img.height
  var s = newFileStream(path, fmWrite)
  defer: s.close()
  s.write "BM"
  s.write uint32(54 + dataSize); s.write 0'u32; s.write 54'u32
  s.write 40'u32; s.write int32(img.width); s.write int32(img.height)
  s.write 1'u16; s.write 24'u16; s.write 0'u32; s.write uint32(dataSize)
  s.write 2835'i32; s.write 2835'i32; s.write 0'u32; s.write 0'u32
  var row = newSeq[uint8](rowSize)
  for y in countdown(img.height - 1, 0):
    for x in 0 ..< img.width:
      let o = (y*img.width + x)*3
      row[3*x] = img.pixels[o+2]; row[3*x+1] = img.pixels[o+1]; row[3*x+2] = img.pixels[o]
    s.writeData(addr row[0], rowSize)

proc writePpm*(img: Image; path: string) =
  # Enregistre au format PPM binaire (P6).
  var s = newFileStream(path, fmWrite)
  defer: s.close()
  s.write "P6\n" & $img.width & " " & $img.height & "\n255\n"
  if img.pixels.len > 0: s.writeData(addr img.pixels[0], img.pixels.len)

proc save*(img: Image; path: string) =
  # Enregistre selon l'extension (.bmp ou .ppm).
  if path.toLowerAscii.endsWith(".ppm"): img.writePpm(path) else: img.writeBmp(path)

proc readBmp*(path: string): Image =
  # Lit un BMP 24 ou 32 bits non compressé.
  let d = readFile(path)
  if d.len < 54 or d[0..1] != "BM": raise newException(IOError, "BMP invalide")
  template u32(o: int): int = int(cast[ptr uint32](unsafeAddr d[o])[])
  template i32(o: int): int = int(cast[ptr int32](unsafeAddr d[o])[])
  let off = u32(10)
  let w = i32(18)
  var h = i32(22)
  let bpp = int(cast[ptr uint16](unsafeAddr d[28])[])
  let comp = u32(30)
  if bpp notin [24, 32] or comp notin [0, 3]: raise newException(IOError, "BMP non pris en charge")
  let topDown = h < 0
  h = abs(h)
  let bytesPP = bpp div 8
  let rowSize = (w * bytesPP + 3) div 4 * 4
  result = Image(width: w, height: h, pixels: newSeq[uint8](w*h*3))
  for y in 0 ..< h:
    let sy = if topDown: y else: h - 1 - y
    for x in 0 ..< w:
      let o = off + sy*rowSize + x*bytesPP
      let p = (y*w + x)*3
      result.pixels[p] = uint8(d[o+2]); result.pixels[p+1] = uint8(d[o+1]); result.pixels[p+2] = uint8(d[o])

proc readPpm*(path: string): Image =
  # Lit un PPM binaire (P6, 8 bits).
  let d = readFile(path)
  var fields: seq[string]
  var i = 0
  while fields.len < 4 and i < d.len:
    if d[i] == '#':
      while i < d.len and d[i] != '\n': inc i
    elif d[i] in Whitespace: inc i
    else:
      var f = ""
      while i < d.len and d[i] notin Whitespace: f.add d[i]; inc i
      fields.add f
  inc i
  if fields.len < 4 or fields[0] != "P6": raise newException(IOError, "PPM P6 attendu")
  let w = parseInt(fields[1])
  let h = parseInt(fields[2])
  result = Image(width: w, height: h, pixels: newSeq[uint8](w*h*3))
  for k in 0 ..< min(w*h*3, d.len - i): result.pixels[k] = uint8(d[i+k])

#
# Canvas (rastérisation avec anticrénelage par suréchantillonnage)
#

proc newCanvas*(w, h: int; bg = rgb(255, 255, 255); ss = 3): Canvas =
  # Surface de dessin de `w`×`h` pixels ; `ss` = suréchantillonnage (1..4).
  result = Canvas(ss: max(1, ss), bw: w * max(1, ss), bh: h * max(1, ss))
  result.img = newImage(w, h, bg)
  result.buf = newSeq[float32](result.bw * result.bh * 3)
  for i in 0 ..< result.bw * result.bh:
    result.buf[3*i] = bg.r; result.buf[3*i+1] = bg.g; result.buf[3*i+2] = bg.b

proc blend(c: Canvas; x, y: int; col: Color) {.inline.} =
  let o = (y*c.bw + x)*3
  let a = col.a
  c.buf[o] = col.r*a + c.buf[o]*(1-a)
  c.buf[o+1] = col.g*a + c.buf[o+1]*(1-a)
  c.buf[o+2] = col.b*a + c.buf[o+2]*(1-a)

proc fillPolygons*(c: Canvas; polys: seq[seq[Point]]; col: Color; evenOdd = false) =
  # Remplit un ensemble de contours (coordonnées en pixels de l'image finale).
  # Règle « nonzero » par défaut, « evenodd » en option.
  if col.a <= 0: return
  let s = c.ss.float
  var minY = Inf
  var maxY = -Inf
  type Edge = tuple[x0, y0, x1, y1: float; dir: int]
  var edges: seq[Edge]
  for poly in polys:
    if poly.len < 2: continue
    for i in 0 ..< poly.len:
      let a = poly[i]
      let b = poly[(i+1) mod poly.len]
      if a.y == b.y: continue
      let e = if a.y < b.y: (a.x*s, a.y*s, b.x*s, b.y*s, 1) else: (b.x*s, b.y*s, a.x*s, a.y*s, -1)
      edges.add e
      minY = min(minY, e[1]); maxY = max(maxY, e[3])
  if edges.len == 0: return
  let y0 = max(0, int(floor(minY)))
  let y1 = min(c.bh - 1, int(ceil(maxY)))
  var xs: seq[(float, int)]
  for y in y0 .. y1:
    let sy = y.float + 0.5
    xs.setLen(0)
    for e in edges:
      if sy >= e.y0 and sy < e.y1:
        let t = (sy - e.y0) / (e.y1 - e.y0)
        xs.add (e.x0 + t*(e.x1 - e.x0), e.dir)
    if xs.len < 2: continue
    xs.sort(proc(a, b: (float, int)): int = cmp(a[0], b[0]))
    var wind = 0
    for i in 0 ..< xs.len - 1:
      wind += (if evenOdd: 1 else: xs[i][1])
      let inside = if evenOdd: (wind mod 2) != 0 else: wind != 0
      if inside:
        let xa = max(0, int(ceil(xs[i][0] - 0.5)))
        let xb = min(c.bw - 1, int(ceil(xs[i+1][0] - 0.5)) - 1)
        for x in xa .. xb: c.blend(x, y, col)

proc fillPolygon*(c: Canvas; pts: seq[Point]; col: Color) =
  c.fillPolygons(@[pts], col)

proc ellipsePoints*(cx, cy, rx, ry: float; n = 0): seq[Point] =
  let k = if n > 0: n else: max(16, int(2*PI*max(rx, ry) / 2))
  for i in 0 ..< k:
    let t = 2*PI*i.float / k.float
    result.add (cx + rx*cos(t), cy + ry*sin(t))

proc fillRect*(c: Canvas; x, y, w, h: float; col: Color) =
  c.fillPolygon(@[(x, y), (x+w, y), (x+w, y+h), (x, y+h)], col)

proc fillCircle*(c: Canvas; cx, cy, r: float; col: Color) =
  c.fillPolygon(ellipsePoints(cx, cy, r, r), col)

proc strokePolyline*(c: Canvas; pts: seq[Point]; width: float; col: Color; closed = false) =
  # Trace un trait d'épaisseur `width` (jointures et extrémités arrondies).
  if pts.len == 0 or width <= 0: return
  let hw = width / 2
  var polys: seq[seq[Point]]
  let n = if closed: pts.len else: pts.len - 1
  for i in 0 ..< n:
    let a = pts[i]
    let b = pts[(i+1) mod pts.len]
    let dx = b.x - a.x
    let dy = b.y - a.y
    let L = sqrt(dx*dx + dy*dy)
    if L < 1e-9: continue
    let nx = -dy / L * hw
    let ny = dx / L * hw
    polys.add @[(a.x+nx, a.y+ny), (b.x+nx, b.y+ny), (b.x-nx, b.y-ny), (a.x-nx, a.y-ny)]
  if hw * c.ss.float >= 1.5:
    for p in pts: polys.add ellipsePoints(p.x, p.y, hw, hw, 12)
  # orientation identique pour tous les morceaux : la règle « nonzero »
  # réalise alors leur union sans trou.
  for p in polys.mitems:
    var area = 0.0
    for i in 0 ..< p.len:
      let q = p[(i+1) mod p.len]
      area += p[i].x*q.y - q.x*p[i].y
    if area < 0: p.reverse()
  c.fillPolygons(polys, col)

proc drawLine*(c: Canvas; x0, y0, x1, y1: float; width: float; col: Color) =
  c.strokePolyline(@[(x0, y0), (x1, y1)], width, col)

proc finish*(c: Canvas): Image =
  # Réduit le tampon suréchantillonné vers l'image finale.
  let s = c.ss
  let inv = 1'f32 / float32(s*s)
  for y in 0 ..< c.img.height:
    for x in 0 ..< c.img.width:
      var r, g, b = 0'f32
      for dy in 0 ..< s:
        for dx in 0 ..< s:
          let o = ((y*s+dy)*c.bw + x*s + dx)*3
          r += c.buf[o]; g += c.buf[o+1]; b += c.buf[o+2]
      let p = (y*c.img.width + x)*3
      c.img.pixels[p] = uint8(clamp(r*inv*255 + 0.5, 0, 255))
      c.img.pixels[p+1] = uint8(clamp(g*inv*255 + 0.5, 0, 255))
      c.img.pixels[p+2] = uint8(clamp(b*inv*255 + 0.5, 0, 255))
  c.img

#
# SVG
#

type
  SvgStyle = object
    fill: Color
    hasFill: bool
    stroke: Color
    hasStroke: bool
    strokeWidth: float
    opacity, fillOpacity, strokeOpacity: float
    evenOdd: bool
    m: Affine

proc identity(): Affine = [1.0, 0, 0, 1, 0, 0]

proc mul(a, b: Affine): Affine =
  # a ∘ b (applique b puis a) [le plus difficile a été de trouver ce #&* de '∘' 😓]
  [a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1],
   a[0]*b[2] + a[2]*b[3], a[1]*b[2] + a[3]*b[3],
   a[0]*b[4] + a[2]*b[5] + a[4], a[1]*b[4] + a[3]*b[5] + a[5]]

proc apply(m: Affine; p: Point): Point =
  (m[0]*p.x + m[2]*p.y + m[4], m[1]*p.x + m[3]*p.y + m[5])

proc scaleFactor(m: Affine): float = sqrt(abs(m[0]*m[3] - m[1]*m[2]))

proc numbers(s: string): seq[float] =
  # Extrait tous les nombres d'une chaîne (gère "1.5.5", "-1-2", "1e3"...).
  var i = 0
  while i < s.len:
    let ch = s[i]
    if ch in {'0'..'9', '.', '-', '+'}:
      var j = i
      if s[j] in {'-', '+'}: inc j
      var dot = false
      while j < s.len and (s[j].isDigit or (s[j] == '.' and not dot)):
        if s[j] == '.': dot = true
        inc j
      if j < s.len and s[j] in {'e', 'E'}:
        var k = j + 1
        if k < s.len and s[k] in {'-', '+'}: inc k
        if k < s.len and s[k].isDigit:
          j = k
          while j < s.len and s[j].isDigit: inc j
      var v: float
      if parseFloat(s[i ..< j], v) > 0: result.add v
      i = max(j, i + 1)
    else: inc i

proc parseTransform(s: string): Affine =
  result = identity()
  var i = 0
  while i < s.len:
    let p = s.find('(', i)
    if p < 0: break
    let q = s.find(')', p)
    if q < 0: break
    let name = s[i ..< p].strip(chars = Whitespace + {','}).toLowerAscii
    let a = numbers(s[p+1 ..< q])
    var m = identity()
    case name
    of "translate":
      m = [1.0, 0, 0, 1, (if a.len > 0: a[0] else: 0.0), (if a.len > 1: a[1] else: 0.0)]
    of "scale":
      let sx = if a.len > 0: a[0] else: 1.0
      let sy = if a.len > 1: a[1] else: sx
      m = [sx, 0, 0, sy, 0, 0]
    of "rotate":
      let t = (if a.len > 0: a[0] else: 0.0) * PI / 180
      m = [cos(t), sin(t), -sin(t), cos(t), 0, 0]
      if a.len >= 3:
        m = mul(mul([1.0, 0, 0, 1, a[1], a[2]], m), [1.0, 0, 0, 1, -a[1], -a[2]])
    of "matrix":
      if a.len >= 6: m = [a[0], a[1], a[2], a[3], a[4], a[5]]
    of "skewx":
      m = [1.0, 0, tan((if a.len > 0: a[0] else: 0.0)*PI/180), 1, 0, 0]
    of "skewy":
      m = [1.0, tan((if a.len > 0: a[0] else: 0.0)*PI/180), 0, 1, 0, 0]
    else: discard
    result = mul(result, m)
    i = q + 1

proc parseAttrs(tag: string): Table[string, string] =
  var i = 0
  while i < tag.len:
    while i < tag.len and tag[i] in Whitespace: inc i
    var name = ""
    while i < tag.len and tag[i] notin Whitespace + {'=', '/', '>'}: name.add tag[i]; inc i
    while i < tag.len and tag[i] in Whitespace: inc i
    if i < tag.len and tag[i] == '=':
      inc i
      while i < tag.len and tag[i] in Whitespace: inc i
      if i < tag.len and tag[i] in {'"', '\''}:
        let qch = tag[i]
        inc i
        var v = ""
        while i < tag.len and tag[i] != qch: v.add tag[i]; inc i
        inc i
        result[name.toLowerAscii] = v
      else:
        var v = ""
        while i < tag.len and tag[i] notin Whitespace: v.add tag[i]; inc i
        result[name.toLowerAscii] = v
    elif name.len == 0: inc i
  if "style" in result:
    for decl in result["style"].split(';'):
      let kv = decl.split(':', 1)
      if kv.len == 2: result[kv[0].strip.toLowerAscii] = kv[1].strip

proc fnum(t: Table[string, string]; k: string; d = 0.0): float =
  if k notin t: return d
  let n = numbers(t[k])
  if n.len > 0: n[0] else: d

proc arcPoints(p0: Point; rx0, ry0, phiDeg: float; large, sweep: bool; p1: Point): seq[Point] =
  # Approximation d'un arc elliptique SVG par des segments.
  var rx = abs(rx0)
  var ry = abs(ry0)
  if rx < 1e-9 or ry < 1e-9: return @[p1]
  let phi = phiDeg * PI / 180
  let cp = cos(phi)
  let sp = sin(phi)
  let dx = (p0.x - p1.x) / 2
  let dy = (p0.y - p1.y) / 2
  let x1 = cp*dx + sp*dy
  let y1 = -sp*dx + cp*dy
  var lam = (x1*x1)/(rx*rx) + (y1*y1)/(ry*ry)
  if lam > 1: rx *= sqrt(lam); ry *= sqrt(lam)
  var num = rx*rx*ry*ry - rx*rx*y1*y1 - ry*ry*x1*x1
  let den = rx*rx*y1*y1 + ry*ry*x1*x1
  var co = if den > 0: sqrt(max(0.0, num/den)) else: 0.0
  if large == sweep: co = -co
  let cxp = co * rx*y1/ry
  let cyp = -co * ry*x1/rx
  let cx = cp*cxp - sp*cyp + (p0.x + p1.x)/2
  let cy = sp*cxp + cp*cyp + (p0.y + p1.y)/2
  proc ang(ux, uy, vx, vy: float): float =
    let a = arctan2(ux*vy - uy*vx, ux*vx + uy*vy)
    a
  let t1 = ang(1, 0, (x1 - cxp)/rx, (y1 - cyp)/ry)
  var dt = ang((x1 - cxp)/rx, (y1 - cyp)/ry, (-x1 - cxp)/rx, (-y1 - cyp)/ry)
  if not sweep and dt > 0: dt -= 2*PI
  elif sweep and dt < 0: dt += 2*PI
  let n = max(4, int(abs(dt) * max(rx, ry) / 3))
  for i in 1 .. n:
    let t = t1 + dt * i.float / n.float
    result.add (cx + rx*cos(t)*cp - ry*sin(t)*sp, cy + rx*cos(t)*sp + ry*sin(t)*cp)

proc parsePath*(d: string): seq[seq[Point]] =
  # Convertit l'attribut `d` d'un chemin SVG en liste de contours.
  var toks: seq[string]
  var i = 0
  while i < d.len:
    let ch = d[i]
    if ch in {'M','m','L','l','H','h','V','v','C','c','S','s','Q','q','T','t','A','a','Z','z'}:
      toks.add $ch; inc i
    elif ch in {'0'..'9', '.', '-', '+'}:
      var j = i
      if d[j] in {'-', '+'}: inc j
      var dot = false
      while j < d.len and (d[j].isDigit or (d[j] == '.' and not dot)):
        if d[j] == '.': dot = true
        inc j
      if j < d.len and d[j] in {'e', 'E'} and j+1 < d.len and (d[j+1].isDigit or d[j+1] in {'-','+'}):
        inc j
        if d[j] in {'-', '+'}: inc j
        while j < d.len and d[j].isDigit: inc j
      toks.add d[i ..< j]
      i = max(j, i + 1)
    else: inc i
  var cur: seq[Point]
  var pos: Point = (0.0, 0.0)
  var start: Point = (0.0, 0.0)
  var lastCtrl: Point = (0.0, 0.0)
  var lastCmd = ' '
  var cmd = ' '
  var k = 0
  proc num(k: var int): float =
    if k < toks.len:
      try: result = parseFloat(toks[k]) except ValueError: result = 0
      inc k
  proc isNum(s: string): bool = s.len > 0 and s[0] notin {'A'..'Z', 'a'..'z'}
  var outp: seq[seq[Point]]
  template flush() =
    if cur.len > 1: outp.add cur
    cur = @[]
  while k < toks.len:
    if not isNum(toks[k]):
      cmd = toks[k][0]; inc k
    elif cmd == ' ':
      inc k; continue
    let rel = cmd in {'a'..'z'}
    let base = if rel: pos else: (0.0, 0.0)
    case cmd.toUpperAscii
    of 'M':
      flush()
      pos = (base.x + num(k), base.y + num(k))
      start = pos
      cur.add pos
      cmd = if rel: 'l' else: 'L'   # les paires suivantes sont des LineTo
    of 'L':
      pos = (base.x + num(k), base.y + num(k)); cur.add pos
    of 'H':
      pos = ((if rel: pos.x else: 0.0) + num(k), pos.y); cur.add pos
    of 'V':
      pos = (pos.x, (if rel: pos.y else: 0.0) + num(k)); cur.add pos
    of 'C', 'S', 'Q', 'T':
      var c1, c2, p: Point
      let up = cmd.toUpperAscii
      if up == 'C':
        c1 = (base.x + num(k), base.y + num(k))
        c2 = (base.x + num(k), base.y + num(k))
        p = (base.x + num(k), base.y + num(k))
      elif up == 'S':
        c1 = if lastCmd.toUpperAscii in {'C', 'S'}: (2*pos.x - lastCtrl.x, 2*pos.y - lastCtrl.y) else: pos
        c2 = (base.x + num(k), base.y + num(k))
        p = (base.x + num(k), base.y + num(k))
      elif up == 'Q':
        c1 = (base.x + num(k), base.y + num(k))
        p = (base.x + num(k), base.y + num(k))
        c2 = c1
      else:
        c1 = if lastCmd.toUpperAscii in {'Q', 'T'}: (2*pos.x - lastCtrl.x, 2*pos.y - lastCtrl.y) else: pos
        p = (base.x + num(k), base.y + num(k))
        c2 = c1
      if cur.len == 0: cur.add pos
      let n = 16
      for s in 1 .. n:
        let t = s.float / n.float
        let u = 1 - t
        var q: Point
        if up in {'C', 'S'}:
          q = (u*u*u*pos.x + 3*u*u*t*c1.x + 3*u*t*t*c2.x + t*t*t*p.x,
               u*u*u*pos.y + 3*u*u*t*c1.y + 3*u*t*t*c2.y + t*t*t*p.y)
        else:
          q = (u*u*pos.x + 2*u*t*c1.x + t*t*p.x, u*u*pos.y + 2*u*t*c1.y + t*t*p.y)
        cur.add q
      lastCtrl = c2
      pos = p
    of 'A':
      let rx = num(k)
      let ry = num(k)
      let rot = num(k)
      let large = num(k) != 0
      let sweep = num(k) != 0
      let p = (base.x + num(k), base.y + num(k))
      if cur.len == 0: cur.add pos
      for q in arcPoints(pos, rx, ry, rot, large, sweep, p): cur.add q
      pos = p
    of 'Z':
      if cur.len > 0: cur.add start
      flush()
      pos = start
      cur.add pos
    else: inc k
    lastCmd = cmd
  flush()
  result = outp

proc svgSize*(svg: string): (float, float) =
  # Taille déclarée d'un document SVG (width/height ou viewBox).
  let a = svg.find("<svg")
  if a < 0: return (0.0, 0.0)
  let b = svg.find('>', a)
  let at = parseAttrs(svg[a+4 ..< b])
  var w = fnum(at, "width", 0)
  var h = fnum(at, "height", 0)
  if "viewbox" in at:
    let vb = numbers(at["viewbox"])
    if vb.len == 4:
      if w == 0: w = vb[2]
      if h == 0: h = vb[3]
  (w, h)

proc renderSvg*(svg: string; width = 0; height = 0; bg = rgb(255, 255, 255)): Image =
  # Rastérise du code SVG en image. Si `width`/`height` valent 0, la taille
  # déclarée par le SVG est utilisée (défaut 512×512).
  var (sw, sh) = svgSize(svg)
  if sw <= 0: sw = 512
  if sh <= 0: sh = 512
  var W = width
  var H = height
  if W <= 0 and H <= 0: W = int(sw); H = int(sh)
  elif W <= 0: W = int(sh.float * 0 + H.float * sw / sh)
  elif H <= 0: H = int(W.float * sh / sw)
  W = clamp(W, 1, 4096); H = clamp(H, 1, 4096)
  let c = newCanvas(W, H, bg, ss = (if W*H > 1_000_000: 1 else: 3))
  # matrice de base : viewBox -> pixels.
  var root = identity()
  let a0 = svg.find("<svg")
  if a0 >= 0:
    let b0 = svg.find('>', a0)
    let at = parseAttrs(svg[a0+4 ..< b0])
    var vb = @[0.0, 0.0, sw, sh]
    if "viewbox" in at:
      let v = numbers(at["viewbox"])
      if v.len == 4 and v[2] > 0 and v[3] > 0: vb = v
    let sx = W.float / vb[2]
    let sy = H.float / vb[3]
    let s = min(sx, sy)
    let ox = (W.float - vb[2]*s) / 2 - vb[0]*s
    let oy = (H.float - vb[3]*s) / 2 - vb[1]*s
    root = [s, 0, 0, s, ox, oy]
  var stack = @[SvgStyle(fill: rgb(0, 0, 0), hasFill: true, strokeWidth: 1, opacity: 1,
                         fillOpacity: 1, strokeOpacity: 1, m: root)]
  var defsDepth = 0
  var i = 0
  while i < svg.len:
    let lt = svg.find('<', i)
    if lt < 0: break
    if svg.continuesWith("<!--", lt):
      let e = svg.find("-->", lt)
      i = if e < 0: svg.len else: e + 3
      continue
    let gt = svg.find('>', lt)
    if gt < 0: break
    var tag = svg[lt+1 ..< gt]
    i = gt + 1
    if tag.len == 0 or tag[0] in {'?', '!'}: continue
    if tag[0] == '/':
      let nm = tag[1 .. ^1].strip.toLowerAscii
      if nm in ["g", "svg", "a"] and stack.len > 1: stack.setLen(stack.len - 1)
      if nm in ["defs", "clippath", "mask", "pattern", "lineargradient",
                "radialgradient", "symbol", "marker", "style"]: dec defsDepth
      continue
    let selfClose = tag.endsWith("/")
    if selfClose: tag = tag[0 .. ^2]
    var j = 0
    while j < tag.len and tag[j] notin Whitespace: inc j
    let name = tag[0 ..< j].toLowerAscii
    let at = parseAttrs(tag[j .. ^1])
    if name in ["defs", "clippath", "mask", "pattern", "lineargradient",
                "radialgradient", "symbol", "marker", "style"]:
      if not selfClose: inc defsDepth
      continue
    if defsDepth > 0: continue
    var st = stack[^1]
    if "transform" in at: st.m = mul(st.m, parseTransform(at["transform"]))
    if "fill" in at:
      if at["fill"].startsWith("url("):
        st.hasFill = true; st.fill = rgb(128, 128, 128)   # dégradés : gris neutre
      else:
        let (ok, col) = parseColor(at["fill"], st.fill)
        st.hasFill = ok
        if ok: st.fill = col
    if "stroke" in at:
      let (ok, col) = parseColor(at["stroke"], st.stroke)
      st.hasStroke = ok
      if ok: st.stroke = col
    if "stroke-width" in at: st.strokeWidth = fnum(at, "stroke-width", 1)
    if "opacity" in at: st.opacity *= fnum(at, "opacity", 1)
    if "fill-opacity" in at: st.fillOpacity = fnum(at, "fill-opacity", 1)
    if "stroke-opacity" in at: st.strokeOpacity = fnum(at, "stroke-opacity", 1)
    if "fill-rule" in at: st.evenOdd = at["fill-rule"].strip == "evenodd"
    if name in ["g", "svg", "a"]:
      if not selfClose:
        if name == "svg" and stack.len == 1: discard   # racine déjà gérée
        stack.add st
      continue
    var polys: seq[seq[Point]]
    var closed = true
    case name
    of "rect":
      let x = fnum(at, "x"); let y = fnum(at, "y")
      let w = fnum(at, "width"); let h = fnum(at, "height")
      var rx = fnum(at, "rx", -1); var ry = fnum(at, "ry", -1)
      if rx < 0: rx = max(0.0, ry)
      if ry < 0: ry = rx
      rx = min(rx, w/2); ry = min(ry, h/2)
      if rx > 0:
        var p: seq[Point]
        for (cx, cy, a0) in [(x+w-rx, y+ry, -PI/2), (x+w-rx, y+h-ry, 0.0),
                             (x+rx, y+h-ry, PI/2), (x+rx, y+ry, PI)]:
          for s in 0 .. 8:
            let t = a0 + PI/2 * s.float / 8
            p.add (cx + rx*cos(t), cy + ry*sin(t))
        polys.add p
      else:
        polys.add @[(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    of "circle":
      let r = fnum(at, "r")
      polys.add ellipsePoints(fnum(at, "cx"), fnum(at, "cy"), r, r, 64)
    of "ellipse":
      polys.add ellipsePoints(fnum(at, "cx"), fnum(at, "cy"), fnum(at, "rx"), fnum(at, "ry"), 64)
    of "line":
      polys.add @[(fnum(at, "x1"), fnum(at, "y1")), (fnum(at, "x2"), fnum(at, "y2"))]
      closed = false
    of "polyline", "polygon":
      let n = numbers(at.getOrDefault("points"))
      var p: seq[Point]
      for k in 0 ..< n.len div 2: p.add (n[2*k], n[2*k+1])
      polys.add p
      closed = name == "polygon"
    of "path":
      polys = parsePath(at.getOrDefault("d"))
      let d = at.getOrDefault("d")
      closed = d.contains('Z') or d.contains('z')
    else:
      continue
    # transformation vers les pixels.
    for p in polys.mitems:
      for q in p.mitems: q = st.m.apply(q)
    if st.hasFill and name notin ["line", "polyline"] or (name == "polyline" and st.hasFill and "fill" in at):
      var col = st.fill
      col.a *= float32(st.opacity * st.fillOpacity)
      c.fillPolygons(polys, col, st.evenOdd)
    if st.hasStroke:
      var col = st.stroke
      col.a *= float32(st.opacity * st.strokeOpacity)
      let w = st.strokeWidth * scaleFactor(st.m)
      for p in polys:
        c.strokePolyline(p, w, col, closed and name != "path" or (name == "path" and p.len > 2 and
                         p[0] == p[^1]))
  c.finish()

proc extractSvg*(text: string): string =
  # Extrait le premier bloc `<svg ...>...</svg>` d'un texte (réponse de LLM).
  let a = text.find("<svg")
  if a < 0: return ""
  let b = text.rfind("</svg>")
  if b < a: return text[a .. ^1] & "</svg>"
  text[a ..< b + 6]
