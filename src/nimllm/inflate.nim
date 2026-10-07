# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026 (adaptation)
# Décodage DEFLATE par codes de Huffman canoniques, structure inspirée de puff.c (distribution zlib)
# © Mark Adker 2002-2013 - LICENSES/Zlib-puff.txt
# nimllm/inflate — décompression DEFLATE / zlib (RFC 1950/1951) en pur Nim.
# Utilisée pour lire les images PNG et le texte des PDF joints.

type
  InflateError* = object of CatchableError
  BitReader = object
    data: ptr UncheckedArray[uint8]
    len, pos: int
    bitBuf: uint32
    bitCnt: int
  Huffman = object
    counts: array[16, int]
    symbols: seq[int]

proc need(br: var BitReader; n: int) =
  while br.bitCnt < n:
    if br.pos >= br.len: raise newException(InflateError, "données compressées tronquées")
    br.bitBuf = br.bitBuf or (uint32(br.data[br.pos]) shl br.bitCnt)
    inc br.pos
    br.bitCnt += 8

proc bits(br: var BitReader; n: int): int =
  if n == 0: return 0
  br.need(n)
  result = int(br.bitBuf and ((1'u32 shl n) - 1))
  br.bitBuf = br.bitBuf shr n
  br.bitCnt -= n

proc build(lengths: openArray[int]): Huffman =
  result.symbols = newSeq[int](lengths.len)
  for l in lengths: inc result.counts[l]
  result.counts[0] = 0
  var offs: array[16, int]
  for i in 1 ..< 16: offs[i] = offs[i-1] + result.counts[i-1]
  for s, l in lengths:
    if l != 0:
      result.symbols[offs[l]] = s
      inc offs[l]

proc decodeSym(br: var BitReader; h: Huffman): int =
  var code, first, index = 0
  for len in 1 .. 15:
    code = code or br.bits(1)
    let count = h.counts[len]
    if code - count < first:
      return h.symbols[index + (code - first)]
    index += count
    first += count
    first = first shl 1
    code = code shl 1
  raise newException(InflateError, "code de Huffman invalide")

const
  lenBase = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
             67, 83, 99, 115, 131, 163, 195, 227, 258]
  lenExtra = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
  distBase = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
              1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577]
  distExtra = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
               11, 11, 12, 12, 13, 13]

proc inflateRaw*(data: openArray[uint8]; maxOut = 512 * 1024 * 1024): seq[uint8] =
  # Décompresse un flux DEFLATE brut.
  if data.len == 0: return
  var br = BitReader(data: cast[ptr UncheckedArray[uint8]](unsafeAddr data[0]), len: data.len)
  var fixedLit, fixedDist: Huffman
  block:
    var l = newSeq[int](288)
    for i in 0 ..< 144: l[i] = 8
    for i in 144 ..< 256: l[i] = 9
    for i in 256 ..< 280: l[i] = 7
    for i in 280 ..< 288: l[i] = 8
    fixedLit = build(l)
    var dl = newSeq[int](30)
    for x in dl.mitems: x = 5
    fixedDist = build(dl)
  while true:
    let final = br.bits(1)
    let typ = br.bits(2)
    case typ
    of 0:
      br.bitBuf = 0; br.bitCnt = 0
      if br.pos + 4 > br.len: raise newException(InflateError, "bloc stocké tronqué")
      let n = int(br.data[br.pos]) or (int(br.data[br.pos+1]) shl 8)
      br.pos += 4
      if br.pos + n > br.len: raise newException(InflateError, "bloc stocké tronqué")
      for i in 0 ..< n: result.add br.data[br.pos + i]
      br.pos += n
    of 1, 2:
      var lit, dist: Huffman
      if typ == 1:
        lit = fixedLit; dist = fixedDist
      else:
        let hlit = br.bits(5) + 257
        let hdist = br.bits(5) + 1
        let hclen = br.bits(4) + 4
        const order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
        var cl = newSeq[int](19)
        for i in 0 ..< hclen: cl[order[i]] = br.bits(3)
        let clh = build(cl)
        var lens = newSeq[int](hlit + hdist)
        var i = 0
        while i < hlit + hdist:
          let sym = br.decodeSym(clh)
          if sym < 16:
            lens[i] = sym; inc i
          else:
            var rep, val = 0
            if sym == 16:
              if i == 0: raise newException(InflateError, "répétition invalide")
              val = lens[i-1]; rep = 3 + br.bits(2)
            elif sym == 17: rep = 3 + br.bits(3)
            else: rep = 11 + br.bits(7)
            for _ in 0 ..< rep:
              if i < lens.len: lens[i] = val
              inc i
        lit = build(lens[0 ..< hlit])
        dist = build(lens[hlit .. ^1])
      while true:
        let sym = br.decodeSym(lit)
        if sym < 256:
          result.add uint8(sym)
        elif sym == 256:
          break
        else:
          let li = sym - 257
          if li >= lenBase.len: raise newException(InflateError, "longueur invalide")
          let length = lenBase[li] + br.bits(lenExtra[li])
          let ds = br.decodeSym(dist)
          if ds >= distBase.len: raise newException(InflateError, "distance invalide")
          let d = distBase[ds] + br.bits(distExtra[ds])
          if d > result.len: raise newException(InflateError, "distance hors limites")
          let start = result.len - d
          for k in 0 ..< length: result.add result[start + k]
        if result.len > maxOut: raise newException(InflateError, "sortie trop volumineuse")
    else:
      raise newException(InflateError, "type de bloc invalide")
    if final == 1: break

proc zlibDecompress*(data: openArray[uint8]): seq[uint8] =
  # Décompresse un flux zlib (en-tête de 2 octets + DEFLATE + Adler-32).
  if data.len < 2: raise newException(InflateError, "flux zlib trop court")
  if (data[0] and 0x0F) != 8:
    # pas d'en-tête zlib : on tente un flux brut.
    return inflateRaw(data)
  inflateRaw(data.toOpenArray(2, data.high))

proc zlibDecompress*(data: string): string =
  if data.len == 0: return ""
  let r = zlibDecompress(data.toOpenArrayByte(0, data.high))
  result = newString(r.len)
  if r.len > 0: copyMem(addr result[0], unsafeAddr r[0], r.len)
