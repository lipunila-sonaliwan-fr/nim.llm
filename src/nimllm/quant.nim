# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026 (adaptation)
# Portions dérivées de ggml / llama.cpp
# The ggml authors (© 2023-2026) - LICENSES/MIT-ggml.txt
# nimllm/quant — formats de poids GGML (F32, F16, BF16, Q4_0, Q4_1, Q5_0,
# Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) : déquantification,
# quantification et produits scalaires.
#
# Les blocs suivent exactement la disposition binaire de llama.cpp/ggml,
# ce qui permet de lire et d'écrire des fichiers GGUF compatibles.

import std/[math, strutils]

type
  GgmlType* = enum
    gtF32 = 0, gtF16 = 1, gtQ4_0 = 2, gtQ4_1 = 3,
    gtQ5_0 = 6, gtQ5_1 = 7, gtQ8_0 = 8, gtQ8_1 = 9,
    gtQ2_K = 10, gtQ3_K = 11, gtQ4_K = 12, gtQ5_K = 13, gtQ6_K = 14, gtQ8_K = 15,
    gtI8 = 24, gtI16 = 25, gtI32 = 26, gtI64 = 27, gtF64 = 28,
    gtBF16 = 30

  Bytes* = ptr UncheckedArray[uint8]
  Floats* = ptr UncheckedArray[float32]

  QuantError* = object of CatchableError

const
  QK_K* = 256
  supportedTypes* = {gtF32, gtF16, gtBF16, gtQ4_0, gtQ4_1, gtQ5_0, gtQ5_1,
                     gtQ8_0, gtQ2_K, gtQ3_K, gtQ4_K, gtQ5_K, gtQ6_K, gtQ8_K}
    # Types déquantifiables par nimllm.
  quantizableTypes* = {gtF32, gtF16, gtBF16, gtQ4_0, gtQ4_1, gtQ5_0, gtQ5_1,
                       gtQ8_0, gtQ4_K, gtQ5_K, gtQ6_K}
    # Types que nimllm sait produire (quantification).

proc isValidType*(v: int): bool =
  v in {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 30}

proc blockSize*(t: GgmlType): int =
  # Nombre d'éléments par bloc.
  case t
  of gtF32, gtF16, gtBF16, gtI8, gtI16, gtI32, gtI64, gtF64: 1
  of gtQ4_0, gtQ4_1, gtQ5_0, gtQ5_1, gtQ8_0, gtQ8_1: 32
  of gtQ2_K, gtQ3_K, gtQ4_K, gtQ5_K, gtQ6_K, gtQ8_K: QK_K

proc typeSize*(t: GgmlType): int =
  # Taille en octets d'un bloc.
  case t
  of gtF32, gtI32: 4
  of gtF16, gtBF16, gtI16: 2
  of gtI8: 1
  of gtI64, gtF64: 8
  of gtQ4_0: 18
  of gtQ4_1: 20
  of gtQ5_0: 22
  of gtQ5_1: 24
  of gtQ8_0: 34
  of gtQ8_1: 36
  of gtQ2_K: 84
  of gtQ3_K: 110
  of gtQ4_K: 144
  of gtQ5_K: 176
  of gtQ6_K: 210
  of gtQ8_K: 292

proc rowBytes*(t: GgmlType; n: int): int =
  # Nombre d'octets nécessaires pour stocker `n` valeurs de type `t`.
  if n mod blockSize(t) != 0:
    raise newException(QuantError, "longueur " & $n & " non multiple du bloc " &
      $blockSize(t) & " pour " & $t)
  (n div blockSize(t)) * typeSize(t)

proc bitsPerWeight*(t: GgmlType): float =
  typeSize(t).float * 8.0 / blockSize(t).float

{.push checks: off, boundChecks: off, overflowChecks: off.}
# (les calculs de ce module sont validés ; on retire les vérifications
#  d'exécution pour la vitesse, même en mode -d:release)

#
# Demi-précision
#

proc halfToFloatSlow(h: uint16): float32 =
  let sign = (h.uint32 and 0x8000'u32) shl 16
  var exp = (h.uint32 shr 10) and 0x1F
  var mant = h.uint32 and 0x3FF
  var bits: uint32
  if exp == 0:
    # zéro ou sous-normal : mant * 2^-24
    let v = float32(mant.float * pow(2.0, -24.0))
    return (if sign != 0: -v else: v)
  elif exp == 31:
    bits = sign or 0x7F800000'u32 or (mant shl 13)
  else:
    bits = sign or ((exp + 127 - 15) shl 23) or (mant shl 13)
  cast[float32](bits)

var halfTable: seq[float32]

proc initHalfTable() =
  halfTable = newSeq[float32](65536)
  for i in 0 ..< 65536:
    halfTable[i] = halfToFloatSlow(uint16(i))

initHalfTable()

template f16*(h: uint16): float32 = halfTable[h.int]
  # Convertit un flottant 16 bits IEEE en float32 (table précalculée).

proc halfToFloat*(h: uint16): float32 = halfTable[h.int]

proc floatToHalf*(f: float32): uint16 =
  # Conversion float32 -> float16 avec arrondi au plus proche.
  let x = cast[uint32](f)
  let sign = (x shr 16) and 0x8000
  let exp = int((x shr 23) and 0xFF)
  var mant = x and 0x7FFFFF
  if exp == 0xFF:
    return uint16(sign or 0x7C00 or (if mant != 0: 0x200'u32 else: 0))
  var e = exp - 127 + 15
  if e >= 31:
    return uint16(sign or 0x7C00)
  if e <= 0:
    if e < -10: return uint16(sign)
    mant = mant or 0x800000
    let shift = uint32(14 - e)
    var hm = mant shr shift
    let rem = mant and ((1'u32 shl shift) - 1)
    let half = 1'u32 shl (shift - 1)
    if rem > half or (rem == half and (hm and 1) == 1): inc hm
    return uint16(sign or hm)
  var hm = mant shr 13
  let rem = mant and 0x1FFF
  var r = uint32(sign) or (uint32(e) shl 10) or hm
  if rem > 0x1000 or (rem == 0x1000 and (hm and 1) == 1): inc r
  uint16(r)

proc bf16ToFloat*(h: uint16): float32 {.inline.} = cast[float32](h.uint32 shl 16)
proc floatToBf16*(f: float32): uint16 =
  let x = cast[uint32](f)
  if (x and 0x7FFFFFFF'u32) > 0x7F800000'u32: return uint16((x shr 16) or 64)
  uint16((x + (0x7FFF'u32 + ((x shr 16) and 1))) shr 16)

template rd16(p: Bytes; o: int): uint16 =
  uint16(p[o]) or (uint16(p[o+1]) shl 8)

template rdHalf(p: Bytes; o: int): float32 = halfTable[rd16(p, o).int]

proc wr16(p: Bytes; o: int; v: uint16) {.inline.} =
  p[o] = uint8(v and 0xFF); p[o+1] = uint8(v shr 8)

#
# Déquantification d'un bloc
#

proc getScaleMinK4(j: int; q: Bytes; o: int; d, m: var uint8) {.inline.} =
  if j < 4:
    d = q[o+j] and 63
    m = q[o+j+4] and 63
  else:
    d = (q[o+j+4] and 0xF) or ((q[o+j-4] shr 6) shl 4)
    m = (q[o+j+4] shr 4) or ((q[o+j] shr 6) shl 4)

proc dequantBlock(t: GgmlType; b: Bytes; y: Floats) {.inline.} =
  # Déquantifie UN bloc situé en `b` vers `y` (blockSize(t) valeurs).
  case t
  of gtQ4_0:
    let d = rdHalf(b, 0)
    for j in 0 ..< 16:
      let q = b[2+j]
      y[j] = float32(int(q and 0xF) - 8) * d
      y[j+16] = float32(int(q shr 4) - 8) * d
  of gtQ4_1:
    let d = rdHalf(b, 0)
    let m = rdHalf(b, 2)
    for j in 0 ..< 16:
      let q = b[4+j]
      y[j] = float32(q and 0xF) * d + m
      y[j+16] = float32(q shr 4) * d + m
  of gtQ5_0:
    let d = rdHalf(b, 0)
    let qh = uint32(b[2]) or (uint32(b[3]) shl 8) or (uint32(b[4]) shl 16) or (uint32(b[5]) shl 24)
    for j in 0 ..< 16:
      let xh0 = ((qh shr j) shl 4) and 0x10
      let xh1 = (qh shr (j + 12)) and 0x10
      let q = b[6+j]
      y[j] = float32(int(uint32(q and 0xF) or xh0) - 16) * d
      y[j+16] = float32(int(uint32(q shr 4) or xh1) - 16) * d
  of gtQ5_1:
    let d = rdHalf(b, 0)
    let m = rdHalf(b, 2)
    let qh = uint32(b[4]) or (uint32(b[5]) shl 8) or (uint32(b[6]) shl 16) or (uint32(b[7]) shl 24)
    for j in 0 ..< 16:
      let xh0 = ((qh shr j) shl 4) and 0x10
      let xh1 = (qh shr (j + 12)) and 0x10
      let q = b[8+j]
      y[j] = float32(uint32(q and 0xF) or xh0) * d + m
      y[j+16] = float32(uint32(q shr 4) or xh1) * d + m
  of gtQ8_0:
    let d = rdHalf(b, 0)
    for j in 0 ..< 32:
      y[j] = float32(cast[int8](b[2+j])) * d
  of gtQ8_1:
    let d = rdHalf(b, 0)
    for j in 0 ..< 32:
      y[j] = float32(cast[int8](b[4+j])) * d
  of gtQ2_K:
    # scales[16] qs[64] d dmin
    let d = rdHalf(b, 80)
    let mn = rdHalf(b, 82)
    var isc = 0
    var yo = 0
    var qo = 16
    for n in 0 ..< 2:
      var shift = 0
      for j in 0 ..< 4:
        var sc = b[isc]; inc isc
        var dl = d * float32(sc and 0xF)
        var ml = mn * float32(sc shr 4)
        for l in 0 ..< 16:
          y[yo] = dl * float32((b[qo+l] shr shift) and 3) - ml; inc yo
        sc = b[isc]; inc isc
        dl = d * float32(sc and 0xF)
        ml = mn * float32(sc shr 4)
        for l in 0 ..< 16:
          y[yo] = dl * float32((b[qo+l+16] shr shift) and 3) - ml; inc yo
        shift += 2
      qo += 32
  of gtQ3_K:
    # hmask[32] qs[64] scales[12] d
    let dAll = rdHalf(b, 108)
    var aux: array[4, uint32]
    copyMem(addr aux[0], addr b[96], 12)
    const kmask1 = 0x03030303'u32
    const kmask2 = 0x0f0f0f0f'u32
    let tmp = aux[2]
    aux[2] = ((aux[0] shr 4) and kmask2) or (((tmp shr 4) and kmask1) shl 4)
    aux[3] = ((aux[1] shr 4) and kmask2) or (((tmp shr 6) and kmask1) shl 4)
    aux[0] = (aux[0] and kmask2) or (((tmp shr 0) and kmask1) shl 4)
    aux[1] = (aux[1] and kmask2) or (((tmp shr 2) and kmask1) shl 4)
    let scales = cast[ptr UncheckedArray[int8]](addr aux[0])
    var m = 1'u8
    var isc = 0
    var yo = 0
    var qo = 32
    for n in 0 ..< 2:
      var shift = 0
      for j in 0 ..< 4:
        var dl = dAll * float32(int(scales[isc]) - 32); inc isc
        for l in 0 ..< 16:
          let hv = if (b[l] and m) != 0: 0 else: 4
          y[yo] = dl * float32(int((b[qo+l] shr shift) and 3) - hv); inc yo
        dl = dAll * float32(int(scales[isc]) - 32); inc isc
        for l in 0 ..< 16:
          let hv = if (b[l+16] and m) != 0: 0 else: 4
          y[yo] = dl * float32(int((b[qo+l+16] shr shift) and 3) - hv); inc yo
        shift += 2
        m = m shl 1
      qo += 32
  of gtQ4_K:
    # d dmin scales[12] qs[128]
    let d = rdHalf(b, 0)
    let mn = rdHalf(b, 2)
    var isc = 0
    var qo = 16
    var yo = 0
    for jj in 0 ..< 4:
      var sc, mm: uint8
      getScaleMinK4(isc, b, 4, sc, mm)
      let d1 = d * sc.float32
      let m1 = mn * mm.float32
      getScaleMinK4(isc + 1, b, 4, sc, mm)
      let d2 = d * sc.float32
      let m2 = mn * mm.float32
      for l in 0 ..< 32: y[yo + l] = d1 * float32(b[qo+l] and 0xF) - m1
      for l in 0 ..< 32: y[yo + 32 + l] = d2 * float32(b[qo+l] shr 4) - m2
      qo += 32; yo += 64; isc += 2
  of gtQ5_K:
    # d dmin scales[12] qh[32] qs[128]
    let d = rdHalf(b, 0)
    let mn = rdHalf(b, 2)
    var isc = 0
    var qo = 48
    var yo = 0
    var u1 = 1'u8
    var u2 = 2'u8
    for jj in 0 ..< 4:
      var sc, mm: uint8
      getScaleMinK4(isc, b, 4, sc, mm)
      let d1 = d * sc.float32
      let m1 = mn * mm.float32
      getScaleMinK4(isc + 1, b, 4, sc, mm)
      let d2 = d * sc.float32
      let m2 = mn * mm.float32
      for l in 0 ..< 32:
        let h = if (b[16+l] and u1) != 0: 16 else: 0
        y[yo+l] = d1 * float32(int(b[qo+l] and 0xF) + h) - m1
      for l in 0 ..< 32:
        let h = if (b[16+l] and u2) != 0: 16 else: 0
        y[yo+32+l] = d2 * float32(int(b[qo+l] shr 4) + h) - m2
      qo += 32; yo += 64; isc += 2
      u1 = u1 shl 2; u2 = u2 shl 2
  of gtQ6_K:
    # ql[128] qh[64] scales[16](int8) d
    let d = rdHalf(b, 208)
    var qlo = 0
    var qho = 128
    var sco = 192
    var yo = 0
    for n in 0 ..< 2:
      for l in 0 ..< 32:
        let isc = l div 16
        let qh = b[qho+l]
        let q1 = int((b[qlo+l] and 0xF) or (((qh shr 0) and 3) shl 4)) - 32
        let q2 = int((b[qlo+l+32] and 0xF) or (((qh shr 2) and 3) shl 4)) - 32
        let q3 = int((b[qlo+l] shr 4) or (((qh shr 4) and 3) shl 4)) - 32
        let q4 = int((b[qlo+l+32] shr 4) or (((qh shr 6) and 3) shl 4)) - 32
        y[yo+l] = d * float32(cast[int8](b[sco+isc])) * q1.float32
        y[yo+l+32] = d * float32(cast[int8](b[sco+isc+2])) * q2.float32
        y[yo+l+64] = d * float32(cast[int8](b[sco+isc+4])) * q3.float32
        y[yo+l+96] = d * float32(cast[int8](b[sco+isc+6])) * q4.float32
      yo += 128; qlo += 64; qho += 32; sco += 8
  of gtQ8_K:
    let d = cast[ptr float32](b)[]
    for j in 0 ..< QK_K:
      y[j] = float32(cast[int8](b[4+j])) * d
  of gtF32:
    y[0] = cast[ptr float32](b)[]
  of gtF16:
    y[0] = rdHalf(b, 0)
  of gtBF16:
    y[0] = bf16ToFloat(rd16(b, 0))
  else:
    raise newException(QuantError, "type non supporté : " & $t)

proc dequantRow*(t: GgmlType; src: pointer; dst: ptr float32; n: int) =
  ## Déquantifie `n` valeurs depuis `src` (format `t`) vers `dst` (float32).
  let s = cast[Bytes](src)
  let y = cast[Floats](dst)
  case t
  of gtF32:
    copyMem(dst, src, n * 4)
  of gtF16:
    for i in 0 ..< n: y[i] = halfTable[rd16(s, 2*i).int]
  of gtBF16:
    for i in 0 ..< n: y[i] = bf16ToFloat(rd16(s, 2*i))
  else:
    let bs = blockSize(t)
    let ts = typeSize(t)
    let nb = n div bs
    for i in 0 ..< nb:
      dequantBlock(t, cast[Bytes](addr s[i*ts]), cast[Floats](addr y[i*bs]))

proc dequantRow*(t: GgmlType; src: pointer; n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0: dequantRow(t, src, addr result[0], n)

#
# Produits scalaires
#

proc dotF32*(a, b: ptr float32; n: int): float32 {.inline.} =
  # Produit scalaire float32 (4 accumulateurs pour aider la vectorisation).
  let x = cast[Floats](a)
  let y = cast[Floats](b)
  var s0, s1, s2, s3, s4, s5, s6, s7: float32
  var i = 0
  while i + 8 <= n:
    s0 += x[i]*y[i]; s1 += x[i+1]*y[i+1]; s2 += x[i+2]*y[i+2]; s3 += x[i+3]*y[i+3]
    s4 += x[i+4]*y[i+4]; s5 += x[i+5]*y[i+5]; s6 += x[i+6]*y[i+6]; s7 += x[i+7]*y[i+7]
    i += 8
  while i < n:
    s0 += x[i]*y[i]; inc i
  (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)

proc dotQ8_0(s: Bytes; x: Floats; n: int): float32 =
  var acc = 0'f32
  let nb = n div 32
  for i in 0 ..< nb:
    let o = i * 34
    let d = rdHalf(s, o)
    var s0, s1, s2, s3: float32
    let xo = i * 32
    for j in countup(0, 31, 4):
      s0 += float32(cast[int8](s[o+2+j])) * x[xo+j]
      s1 += float32(cast[int8](s[o+3+j])) * x[xo+j+1]
      s2 += float32(cast[int8](s[o+4+j])) * x[xo+j+2]
      s3 += float32(cast[int8](s[o+5+j])) * x[xo+j+3]
    acc += d * ((s0 + s1) + (s2 + s3))
  acc

proc dotQ4_0(s: Bytes; x: Floats; n: int): float32 =
  var acc = 0'f32
  let nb = n div 32
  for i in 0 ..< nb:
    let o = i * 18
    let d = rdHalf(s, o)
    let xo = i * 32
    var sa, sb: float32
    for j in 0 ..< 16:
      let q = s[o+2+j]
      sa += float32(int(q and 0xF) - 8) * x[xo+j]
      sb += float32(int(q shr 4) - 8) * x[xo+j+16]
    acc += d * (sa + sb)
  acc

proc dotRow*(t: GgmlType; src: pointer; x: ptr float32; n: int; tmp: ptr float32): float32 =
  # Produit scalaire entre une ligne quantifiée et un vecteur float32.
  # `tmp` doit pouvoir contenir `n` floats (utilisé pour les formats génériques).
  case t
  of gtF32: dotF32(cast[ptr float32](src), x, n)
  of gtQ8_0: dotQ8_0(cast[Bytes](src), cast[Floats](x), n)
  of gtQ4_0: dotQ4_0(cast[Bytes](src), cast[Floats](x), n)
  else:
    dequantRow(t, src, tmp, n)
    dotF32(tmp, x, n)

#
# Quantification (format de référence simple, compatible ggml)
#

proc nearestInt(f: float32): int {.inline.} = int(round(f))

proc quantizeBlock(t: GgmlType; x: Floats; b: Bytes) =
  case t
  of gtQ8_0:
    var amax = 0'f32
    for j in 0 ..< 32: amax = max(amax, abs(x[j]))
    let d = amax / 127'f32
    let id = if d != 0: 1'f32 / d else: 0'f32
    wr16(b, 0, floatToHalf(d))
    for j in 0 ..< 32:
      b[2+j] = cast[uint8](int8(clamp(nearestInt(x[j] * id), -127, 127)))
  of gtQ4_0:
    var amax = 0'f32
    var mx = 0'f32
    for j in 0 ..< 32:
      if abs(x[j]) > amax: amax = abs(x[j]); mx = x[j]
    let d = mx / -8'f32
    let id = if d != 0: 1'f32 / d else: 0'f32
    wr16(b, 0, floatToHalf(d))
    for j in 0 ..< 16:
      let x0 = x[j] * id
      let x1 = x[16+j] * id
      let q0 = uint8(min(15, int(x0 + 8.5'f32)))
      let q1 = uint8(min(15, int(x1 + 8.5'f32)))
      b[2+j] = q0 or (q1 shl 4)
  of gtQ4_1:
    var mn = Inf.float32
    var mx = -Inf.float32
    for j in 0 ..< 32: mn = min(mn, x[j]); mx = max(mx, x[j])
    let d = (mx - mn) / 15'f32
    let id = if d != 0: 1'f32 / d else: 0'f32
    wr16(b, 0, floatToHalf(d)); wr16(b, 2, floatToHalf(mn))
    for j in 0 ..< 16:
      let q0 = uint8(min(15, int((x[j] - mn) * id + 0.5'f32)))
      let q1 = uint8(min(15, int((x[16+j] - mn) * id + 0.5'f32)))
      b[4+j] = q0 or (q1 shl 4)
  of gtQ5_0:
    var amax = 0'f32
    var mx = 0'f32
    for j in 0 ..< 32:
      if abs(x[j]) > amax: amax = abs(x[j]); mx = x[j]
    let d = mx / -16'f32
    let id = if d != 0: 1'f32 / d else: 0'f32
    wr16(b, 0, floatToHalf(d))
    var qh = 0'u32
    for j in 0 ..< 16:
      let q0 = uint8(min(31, int(x[j] * id + 16.5'f32)))
      let q1 = uint8(min(31, int(x[16+j] * id + 16.5'f32)))
      b[6+j] = (q0 and 0xF) or ((q1 and 0xF) shl 4)
      qh = qh or (uint32((q0 and 0x10) shr 4) shl j)
      qh = qh or (uint32((q1 and 0x10) shr 4) shl (j + 16))
    for k in 0 ..< 4: b[2+k] = uint8((qh shr (8*k)) and 0xFF)
  of gtQ5_1:
    var mn = Inf.float32
    var mx = -Inf.float32
    for j in 0 ..< 32: mn = min(mn, x[j]); mx = max(mx, x[j])
    let d = (mx - mn) / 31'f32
    let id = if d != 0: 1'f32 / d else: 0'f32
    wr16(b, 0, floatToHalf(d)); wr16(b, 2, floatToHalf(mn))
    var qh = 0'u32
    for j in 0 ..< 16:
      let q0 = uint8(min(31, int((x[j] - mn) * id + 0.5'f32)))
      let q1 = uint8(min(31, int((x[16+j] - mn) * id + 0.5'f32)))
      b[8+j] = (q0 and 0xF) or ((q1 and 0xF) shl 4)
      qh = qh or (uint32((q0 and 0x10) shr 4) shl j)
      qh = qh or (uint32((q1 and 0x10) shr 4) shl (j + 16))
    for k in 0 ..< 4: b[4+k] = uint8((qh shr (8*k)) and 0xFF)
  of gtQ4_K, gtQ5_K:
    # 8 sous-blocs de 32 : échelle et minimum sur 6 bits.
    let nmax = if t == gtQ4_K: 15'f32 else: 31'f32
    var scales, mins: array[8, float32]
    for s in 0 ..< 8:
      var mn = 0'f32             # le minimum est toujours <= 0 (comme ggml).
      var mx = -Inf.float32
      for l in 0 ..< 32:
        mn = min(mn, x[32*s+l]); mx = max(mx, x[32*s+l])
      scales[s] = max(0'f32, (mx - mn) / nmax)
      mins[s] = -mn
    var maxSc = 0'f32
    var maxMn = 0'f32
    for s in 0 ..< 8: maxSc = max(maxSc, scales[s]); maxMn = max(maxMn, mins[s])
    let invSc = if maxSc > 0: 63'f32 / maxSc else: 0'f32
    let invMn = if maxMn > 0: 63'f32 / maxMn else: 0'f32
    var ls, lm: array[8, uint8]
    for s in 0 ..< 8:
      ls[s] = uint8(min(63, nearestInt(invSc * scales[s])))
      lm[s] = uint8(min(63, nearestInt(invMn * mins[s])))
    let d = maxSc / 63'f32
    let dmin = maxMn / 63'f32
    wr16(b, 0, floatToHalf(d)); wr16(b, 2, floatToHalf(dmin))
    # empaquetage des échelles (12 octets).
    for j in 0 ..< 8:
      if j < 4:
        b[4+j] = ls[j]
        b[4+j+4] = lm[j]
      else:
        b[4+j+4] = (ls[j] and 0xF) or ((lm[j] and 0xF) shl 4)
        b[4+j-4] = b[4+j-4] or ((ls[j] shr 4) shl 6)
        b[4+j] = b[4+j] or ((lm[j] shr 4) shl 6)
    let dh = halfToFloat(floatToHalf(d))
    let mh = halfToFloat(floatToHalf(dmin))
    var L: array[256, uint8]
    for s in 0 ..< 8:
      let dd = dh * ls[s].float32
      let dm = mh * lm[s].float32
      for l in 0 ..< 32:
        let v = if dd > 0: nearestInt((x[32*s+l] + dm) / dd) else: 0
        L[32*s+l] = uint8(clamp(v, 0, nmax.int))
    if t == gtQ4_K:
      for j in 0 ..< 4:
        for l in 0 ..< 32:
          b[16 + 32*j + l] = L[64*j + l] or (L[64*j + 32 + l] shl 4)
    else:
      for l in 0 ..< 32: b[16+l] = 0
      var m1 = 1'u8
      var m2 = 2'u8
      for j in 0 ..< 4:
        for l in 0 ..< 32:
          var l1 = L[64*j + l]
          if l1 > 15: l1 -= 16; b[16+l] = b[16+l] or m1
          var l2 = L[64*j + 32 + l]
          if l2 > 15: l2 -= 16; b[16+l] = b[16+l] or m2
          b[48 + 32*j + l] = l1 or (l2 shl 4)
        m1 = m1 shl 2; m2 = m2 shl 2
  of gtQ6_K:
    # échelle positive par groupe de 16 : valeurs entières dans [-31, 31].
    var scales: array[16, float32]
    var maxScale = 0'f32
    for s in 0 ..< 16:
      var amax = 0'f32
      for l in 0 ..< 16: amax = max(amax, abs(x[16*s+l]))
      scales[s] = amax / 31'f32
      maxScale = max(maxScale, scales[s])
    let iscale = if maxScale > 0: 127'f32 / maxScale else: 0'f32
    let d = if maxScale > 0: 1'f32 / iscale else: 0'f32
    wr16(b, 208, floatToHalf(d))
    let dh = halfToFloat(floatToHalf(d))
    var L: array[256, uint8]
    for s in 0 ..< 16:
      let sc = int8(clamp(nearestInt(iscale * scales[s]), -128, 127))
      b[192+s] = cast[uint8](sc)
      let dd = dh * sc.float32
      for l in 0 ..< 16:
        let v = if dd != 0: nearestInt(x[16*s+l] / dd) else: 0
        L[16*s+l] = uint8(clamp(v, -32, 31) + 32)
    for j in 0 ..< 2:
      for l in 0 ..< 32:
        let q1 = L[128*j + l] and 0xF
        let q2 = L[128*j + l + 32] and 0xF
        let q3 = L[128*j + l + 64] and 0xF
        let q4 = L[128*j + l + 96] and 0xF
        b[64*j + l] = q1 or (q3 shl 4)
        b[64*j + l + 32] = q2 or (q4 shl 4)
        b[128 + 32*j + l] = (L[128*j+l] shr 4) or ((L[128*j+l+32] shr 4) shl 2) or
                            ((L[128*j+l+64] shr 4) shl 4) or ((L[128*j+l+96] shr 4) shl 6)
  else:
    raise newException(QuantError, "quantification non supportée : " & $t)

proc quantizeRow*(t: GgmlType; src: ptr float32; dst: pointer; n: int) =
  # Quantifie `n` floats vers le format `t` (écrit `rowBytes(t, n)` octets).
  let x = cast[Floats](src)
  let d = cast[Bytes](dst)
  case t
  of gtF32: copyMem(dst, src, n * 4)
  of gtF16:
    for i in 0 ..< n: wr16(d, 2*i, floatToHalf(x[i]))
  of gtBF16:
    for i in 0 ..< n: wr16(d, 2*i, floatToBf16(x[i]))
  else:
    let bs = blockSize(t)
    let ts = typeSize(t)
    if n mod bs != 0:
      raise newException(QuantError, "longueur non multiple de " & $bs)
    for i in 0 ..< n div bs:
      quantizeBlock(t, cast[Floats](addr x[i*bs]), cast[Bytes](addr d[i*ts]))

proc quantize*(t: GgmlType; data: openArray[float32]): seq[uint8] =
  # Quantifie un tableau complet (sa longueur doit être multiple du bloc).
  result = newSeq[uint8](rowBytes(t, data.len))
  if data.len > 0:
    quantizeRow(t, unsafeAddr data[0], addr result[0], data.len)

{.pop.}

proc parseGgmlType*(s: string): GgmlType =
  # "q4_k", "Q8_0", "f16"... -> GgmlType
  case s.toLowerAscii
  of "f32": gtF32
  of "f16": gtF16
  of "bf16": gtBF16
  of "q4_0": gtQ4_0
  of "q4_1": gtQ4_1
  of "q5_0": gtQ5_0
  of "q5_1": gtQ5_1
  of "q8_0": gtQ8_0
  of "q2_k": gtQ2_K
  of "q3_k": gtQ3_K
  of "q4_k": gtQ4_K
  of "q5_k": gtQ5_K
  of "q6_k": gtQ6_K
  else: raise newException(QuantError, "type inconnu : " & s)

#
# Produits scalaires entiers (activation quantifiée en 8 bits, comme ggml)
#

{.push checks: off, boundChecks: off, overflowChecks: off, rangeChecks: off.}

type
  QAct* = object
    # Vecteur d'activation quantifié en int8 par blocs (32 ou 256 valeurs).
    blockLen*: int
    d*: seq[float32]             # échelle par bloc.
    qs*: seq[int8]
    bsums*: seq[int32]           # sommes par groupe de 16 (blocs de 256).

proc actBlockLen*(t: GgmlType): int =
  # Taille de bloc d'activation adaptée au type de poids (0 = pas de chemin entier).
  case t
  of gtQ4_K, gtQ5_K, gtQ6_K: 256
  of gtQ8_0, gtQ4_0: 32
  else: 0

proc quantizeAct*(x: ptr float32; n, blockLen: int; a: var QAct) =
  # Quantifie `n` floats en int8 (échelle par bloc de `blockLen`).
  let xs = cast[Floats](x)
  let nb = n div blockLen
  a.blockLen = blockLen
  a.d.setLen(nb)
  a.qs.setLen(n)
  if blockLen == 256: a.bsums.setLen(n div 16)
  for b in 0 ..< nb:
    var amax = 0'f32
    for i in 0 ..< blockLen: amax = max(amax, abs(xs[b*blockLen + i]))
    let d = amax / 127'f32
    let id = if d > 0: 1'f32 / d else: 0'f32
    a.d[b] = d
    for i in 0 ..< blockLen:
      let v = xs[b*blockLen + i] * id
      a.qs[b*blockLen + i] = int8(if v >= 0: int(v + 0.5) else: -int(-v + 0.5))
    if blockLen == 256:
      for g in 0 ..< 16:
        var s = 0'i32
        for i in 0 ..< 16: s += a.qs[b*256 + g*16 + i].int32
        a.bsums[b*16 + g] = s

proc dotQ4K(w: Bytes; a: ptr QAct; off, n: int): float32 =
  let nb = n div 256
  var acc = 0'f32
  for ib in 0 ..< nb:
    let b = cast[Bytes](addr w[ib*144])
    let ab = off div 256 + ib
    let q8 = cast[ptr UncheckedArray[int8]](unsafeAddr a.qs[ab*256])
    let d = rdHalf(b, 0)
    let dmin = rdHalf(b, 2)
    var sumi = 0'i32
    var summ = 0'i32
    for j in 0 ..< 4:
      var sc0, m0, sc1, m1: uint8
      getScaleMinK4(2*j, b, 4, sc0, m0)
      getScaleMinK4(2*j + 1, b, 4, sc1, m1)
      var s0 = 0'i32
      var s1 = 0'i32
      let qo = 16 + 32*j
      for l in 0 ..< 32:
        let q = b[qo + l]
        s0 += int32(q and 0xF) * q8[64*j + l].int32
        s1 += int32(q shr 4) * q8[64*j + 32 + l].int32
      sumi += sc0.int32 * s0 + sc1.int32 * s1
      let bs = cast[ptr UncheckedArray[int32]](unsafeAddr a.bsums[ab*16 + 4*j])
      summ += m0.int32 * (bs[0] + bs[1]) + m1.int32 * (bs[2] + bs[3])
    acc += a.d[ab] * (d * sumi.float32 - dmin * summ.float32)
  acc

proc dotQ5K(w: Bytes; a: ptr QAct; off, n: int): float32 =
  let nb = n div 256
  var acc = 0'f32
  for ib in 0 ..< nb:
    let b = cast[Bytes](addr w[ib*176])
    let ab = off div 256 + ib
    let q8 = cast[ptr UncheckedArray[int8]](unsafeAddr a.qs[ab*256])
    let d = rdHalf(b, 0)
    let dmin = rdHalf(b, 2)
    var sumi = 0'i32
    var summ = 0'i32
    var u1 = 1'u8
    var u2 = 2'u8
    for j in 0 ..< 4:
      var sc0, m0, sc1, m1: uint8
      getScaleMinK4(2*j, b, 4, sc0, m0)
      getScaleMinK4(2*j + 1, b, 4, sc1, m1)
      var s0 = 0'i32
      var s1 = 0'i32
      let qo = 48 + 32*j
      for l in 0 ..< 32:
        let q = b[qo + l]
        let h = b[16 + l]
        let v0 = int32(q and 0xF) + (if (h and u1) != 0: 16'i32 else: 0'i32)
        let v1 = int32(q shr 4) + (if (h and u2) != 0: 16'i32 else: 0'i32)
        s0 += v0 * q8[64*j + l].int32
        s1 += v1 * q8[64*j + 32 + l].int32
      sumi += sc0.int32 * s0 + sc1.int32 * s1
      let bs = cast[ptr UncheckedArray[int32]](unsafeAddr a.bsums[ab*16 + 4*j])
      summ += m0.int32 * (bs[0] + bs[1]) + m1.int32 * (bs[2] + bs[3])
      u1 = u1 shl 2; u2 = u2 shl 2
    acc += a.d[ab] * (d * sumi.float32 - dmin * summ.float32)
  acc

proc dotQ6K(w: Bytes; a: ptr QAct; off, n: int): float32 =
  let nb = n div 256
  var acc = 0'f32
  for ib in 0 ..< nb:
    let b = cast[Bytes](addr w[ib*210])
    let ab = off div 256 + ib
    let q8 = cast[ptr UncheckedArray[int8]](unsafeAddr a.qs[ab*256])
    let d = rdHalf(b, 208)
    let sc = cast[ptr UncheckedArray[int8]](addr b[192])
    var sumi = 0'i32
    for h in 0 ..< 2:
      let ql = 64*h
      let qh = 128 + 32*h
      for half in 0 ..< 2:
        var s0, s1, s2, s3 = 0'i32
        for l in 16*half ..< 16*half + 16:
          let hb = b[qh + l]
          let q1 = int32((b[ql + l] and 0xF) or ((hb and 3) shl 4)) - 32
          let q2 = int32((b[ql + l + 32] and 0xF) or (((hb shr 2) and 3) shl 4)) - 32
          let q3 = int32((b[ql + l] shr 4) or (((hb shr 4) and 3) shl 4)) - 32
          let q4 = int32((b[ql + l + 32] shr 4) or (((hb shr 6) and 3) shl 4)) - 32
          let base = 128*h + l
          s0 += q1 * q8[base].int32
          s1 += q2 * q8[base + 32].int32
          s2 += q3 * q8[base + 64].int32
          s3 += q4 * q8[base + 96].int32
        let so = 8*h + half
        sumi += sc[so].int32 * s0 + sc[so + 2].int32 * s1 + sc[so + 4].int32 * s2 + sc[so + 6].int32 * s3
    acc += a.d[ab] * d * sumi.float32
  acc

proc dotQ8_0i(w: Bytes; a: ptr QAct; off, n: int): float32 =
  let nb = n div 32
  var acc = 0'f32
  for ib in 0 ..< nb:
    let o = ib * 34
    let ab = off div 32 + ib
    let q8 = cast[ptr UncheckedArray[int8]](unsafeAddr a.qs[ab*32])
    var s = 0'i32
    for l in 0 ..< 32: s += cast[int8](w[o + 2 + l]).int32 * q8[l].int32
    acc += rdHalf(w, o) * a.d[ab] * s.float32
  acc

proc dotQ4_0i(w: Bytes; a: ptr QAct; off, n: int): float32 =
  let nb = n div 32
  var acc = 0'f32
  for ib in 0 ..< nb:
    let o = ib * 18
    let ab = off div 32 + ib
    let q8 = cast[ptr UncheckedArray[int8]](unsafeAddr a.qs[ab*32])
    var s = 0'i32
    for l in 0 ..< 16:
      let q = w[o + 2 + l]
      s += (int32(q and 0xF) - 8) * q8[l].int32 + (int32(q shr 4) - 8) * q8[l + 16].int32
    acc += rdHalf(w, o) * a.d[ab] * s.float32
  acc

proc dotRowAct*(t: GgmlType; src: pointer; a: ptr QAct; off, n: int): float32 =
  # Produit scalaire ligne quantifiée × activation quantifiée (éléments
  # `off ..< off+n` de l'activation).
  let w = cast[Bytes](src)
  case t
  of gtQ4_K: dotQ4K(w, a, off, n)
  of gtQ5_K: dotQ5K(w, a, off, n)
  of gtQ6_K: dotQ6K(w, a, off, n)
  of gtQ8_0: dotQ8_0i(w, a, off, n)
  of gtQ4_0: dotQ4_0i(w, a, off, n)
  else: 0

{.pop.}
