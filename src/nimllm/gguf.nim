# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026 (adaptation)
# Portions dérivées de ggml / llama.cpp
# The ggml authors (© 2023-2026) - LICENSES/MIT-ggml.txt
# nimllm/gguf — lecture (mmap) et écriture de fichiers GGUF v2/v3.
#
# GGUF est le format de fichier de llama.cpp : un en-tête de métadonnées
# clé/valeur suivi des descriptions de tenseurs puis des données alignées.

import std/[memfiles, tables, strutils, streams, os]
import quant
export tables

type
  GgufValueKind* = enum
    gvU8 = 0, gvI8 = 1, gvU16 = 2, gvI16 = 3, gvU32 = 4, gvI32 = 5,
    gvF32 = 6, gvBool = 7, gvString = 8, gvArray = 9, gvU64 = 10,
    gvI64 = 11, gvF64 = 12

  GgufValue* = ref object
    # Valeur de métadonnée GGUF (entier, flottant, chaîne, booléen ou tableau).
    case kind*: GgufValueKind
    of gvU8, gvI8, gvU16, gvI16, gvU32, gvI32, gvU64, gvI64: i*: int64
    of gvF32, gvF64: f*: float64
    of gvBool: b*: bool
    of gvString: s*: string
    of gvArray:
      elemKind*: GgufValueKind
      arr*: seq[GgufValue]

  GgufTensorInfo* = object
    name*: string
    dims*: seq[int]              # ne[0] = dimension la plus interne (longueur de ligne).
    typ*: GgmlType
    offset*: int                 # relatif au début de la zone de données.
    data*: pointer               # adresse en mémoire (fichier mappé).

  GgufFile* = ref object
    # Fichier GGUF ouvert (mappé en mémoire, lecture seule).
    path*: string
    version*: int
    kv*: OrderedTable[string, GgufValue]
    tensors*: OrderedTable[string, GgufTensorInfo]
    alignment*: int
    dataOffset*: int
    mf: MemFile
    base: ptr UncheckedArray[uint8]
    size: int

  GgufError* = object of CatchableError

#
# Constructeurs pratiques
#

proc gStr*(s: string): GgufValue = GgufValue(kind: gvString, s: s)
proc gU32*(v: int): GgufValue = GgufValue(kind: gvU32, i: v)
proc gI32*(v: int): GgufValue = GgufValue(kind: gvI32, i: v)
proc gU64*(v: int): GgufValue = GgufValue(kind: gvU64, i: v)
proc gF32*(v: float): GgufValue = GgufValue(kind: gvF32, f: v)
proc gBool*(v: bool): GgufValue = GgufValue(kind: gvBool, b: v)
proc gArrStr*(v: openArray[string]): GgufValue =
  result = GgufValue(kind: gvArray, elemKind: gvString)
  for x in v: result.arr.add gStr(x)
proc gArrI32*(v: openArray[int]): GgufValue =
  result = GgufValue(kind: gvArray, elemKind: gvI32)
  for x in v: result.arr.add gI32(x)
proc gArrF32*(v: openArray[float32]): GgufValue =
  result = GgufValue(kind: gvArray, elemKind: gvF32)
  for x in v: result.arr.add gF32(x.float)

proc `$`*(v: GgufValue): string =
  case v.kind
  of gvString: (if v.s.len > 80: v.s[0..76].escape & "..." else: v.s.escape)
  of gvArray: "[" & $v.elemKind & " x " & $v.arr.len & "]"
  of gvBool: $v.b
  of gvF32, gvF64: $v.f
  else: $v.i

#
# Lecture
#

type Reader = object
  p: ptr UncheckedArray[uint8]
  pos, size: int

proc need(r: var Reader; n: int) =
  if r.pos + n > r.size:
    raise newException(GgufError, "fichier GGUF tronqué")

proc rd[T](r: var Reader): T =
  r.need(sizeof(T))
  copyMem(addr result, addr r.p[r.pos], sizeof(T))
  r.pos += sizeof(T)

proc rdStr(r: var Reader): string =
  let n = int(rd[uint64](r))
  r.need(n)
  result = newString(n)
  if n > 0: copyMem(addr result[0], addr r.p[r.pos], n)
  r.pos += n

proc rdValue(r: var Reader; k: GgufValueKind): GgufValue =
  case k
  of gvU8: GgufValue(kind: k, i: rd[uint8](r).int64)
  of gvI8: GgufValue(kind: k, i: rd[int8](r).int64)
  of gvU16: GgufValue(kind: k, i: rd[uint16](r).int64)
  of gvI16: GgufValue(kind: k, i: rd[int16](r).int64)
  of gvU32: GgufValue(kind: k, i: rd[uint32](r).int64)
  of gvI32: GgufValue(kind: k, i: rd[int32](r).int64)
  of gvU64: GgufValue(kind: k, i: cast[int64](rd[uint64](r)))
  of gvI64: GgufValue(kind: k, i: rd[int64](r))
  of gvF32: GgufValue(kind: k, f: rd[float32](r).float64)
  of gvF64: GgufValue(kind: k, f: rd[float64](r))
  of gvBool: GgufValue(kind: k, b: rd[uint8](r) != 0)
  of gvString: GgufValue(kind: k, s: rdStr(r))
  of gvArray:
    let ek = rd[uint32](r)
    if ek > 12: raise newException(GgufError, "type de tableau invalide")
    let n = int(rd[uint64](r))
    var v = GgufValue(kind: gvArray, elemKind: GgufValueKind(ek))
    v.arr = newSeqOfCap[GgufValue](n)
    for _ in 0 ..< n: v.arr.add rdValue(r, GgufValueKind(ek))
    v

proc openGguf*(path: string): GgufFile =
  # Ouvre un fichier GGUF en le mappant en mémoire (aucune copie des poids).
  if not fileExists(path):
    raise newException(GgufError, "fichier introuvable : " & path)
  result = GgufFile(path: path, alignment: 32)
  result.mf = memfiles.open(path, mode = fmRead)
  result.base = cast[ptr UncheckedArray[uint8]](result.mf.mem)
  result.size = result.mf.size
  var r = Reader(p: result.base, pos: 0, size: result.size)
  let magic = rd[uint32](r)
  if magic != 0x46554747'u32:    # "GGUF" !
    raise newException(GgufError, "ce n'est pas un fichier GGUF : " & path)
  result.version = int(rd[uint32](r))
  if result.version < 2:
    raise newException(GgufError, "version GGUF trop ancienne : " & $result.version)
  let nt = int(rd[uint64](r))
  let nkv = int(rd[uint64](r))
  for _ in 0 ..< nkv:
    let key = rdStr(r)
    let k = rd[uint32](r)
    if k > 12: raise newException(GgufError, "type de valeur invalide pour " & key)
    result.kv[key] = rdValue(r, GgufValueKind(k))
  if "general.alignment" in result.kv:
    result.alignment = int(result.kv["general.alignment"].i)
  var infos: seq[GgufTensorInfo]
  for _ in 0 ..< nt:
    var ti: GgufTensorInfo
    ti.name = rdStr(r)
    let nd = int(rd[uint32](r))
    for _ in 0 ..< nd: ti.dims.add int(rd[uint64](r))
    let t = int(rd[uint32](r))
    if not isValidType(t):
      raise newException(GgufError, "type de tenseur inconnu (" & $t & ") pour " & ti.name &
        " — formats IQ* non pris en charge")
    {.push warning[HoleEnumConv]: off.}
    ti.typ = GgmlType(t)
    {.pop.}
    ti.offset = int(rd[uint64](r))
    infos.add ti
  let a = result.alignment
  result.dataOffset = (r.pos + a - 1) div a * a
  for ti in infos.mitems:
    ti.data = addr result.base[result.dataOffset + ti.offset]
    result.tensors[ti.name] = ti

proc close*(g: GgufFile) =
  # Libère le mappage mémoire.
  if g.base != nil:
    g.mf.close()
    g.base = nil

proc has*(g: GgufFile; key: string): bool = key in g.kv

proc getInt*(g: GgufFile; key: string; default = 0): int =
  if key in g.kv:
    let v = g.kv[key]
    case v.kind
    of gvF32, gvF64: int(v.f)
    of gvBool: ord(v.b)
    of gvString, gvArray: default
    else: int(v.i)
  else: default

proc getFloat*(g: GgufFile; key: string; default = 0.0): float =
  if key in g.kv:
    let v = g.kv[key]
    case v.kind
    of gvF32, gvF64: v.f
    of gvString, gvArray, gvBool: default
    else: v.i.float
  else: default

proc getStr*(g: GgufFile; key: string; default = ""): string =
  if key in g.kv and g.kv[key].kind == gvString: g.kv[key].s else: default

proc getBool*(g: GgufFile; key: string; default = false): bool =
  if key in g.kv and g.kv[key].kind == gvBool: g.kv[key].b else: default

proc getStrArray*(g: GgufFile; key: string): seq[string] =
  if key in g.kv and g.kv[key].kind == gvArray:
    for v in g.kv[key].arr: result.add v.s

proc getFloatArray*(g: GgufFile; key: string): seq[float32] =
  if key in g.kv and g.kv[key].kind == gvArray:
    for v in g.kv[key].arr:
      result.add(if v.kind in {gvF32, gvF64}: v.f.float32 else: v.i.float32)

proc getIntArray*(g: GgufFile; key: string): seq[int] =
  if key in g.kv and g.kv[key].kind == gvArray:
    for v in g.kv[key].arr:
      result.add(if v.kind in {gvF32, gvF64}: int(v.f) else: int(v.i))

proc numElements*(ti: GgufTensorInfo): int =
  result = 1
  for d in ti.dims: result *= d

proc byteSize*(ti: GgufTensorInfo): int =
  rowBytes(ti.typ, ti.dims[0]) * (ti.numElements div ti.dims[0])

proc tensorF32*(g: GgufFile; name: string): seq[float32] =
  # Retourne le tenseur `name` entièrement déquantifié en float32.
  if name notin g.tensors:
    raise newException(GgufError, "tenseur absent : " & name)
  let ti = g.tensors[name]
  let n = ti.numElements
  result = newSeq[float32](n)
  let rowLen = ti.dims[0]
  let rb = rowBytes(ti.typ, rowLen)
  let p = cast[ptr UncheckedArray[uint8]](ti.data)
  for r in 0 ..< n div rowLen:
    dequantRow(ti.typ, addr p[r*rb], addr result[r*rowLen], rowLen)

proc describe*(g: GgufFile; maxTensors = 10): string =
  # Résumé lisible (métadonnées + premiers tenseurs).
  result.add "GGUF v" & $g.version & " — " & $g.kv.len & " métadonnées, " &
    $g.tensors.len & " tenseurs\n"
  for k, v in g.kv:
    result.add "  " & k & " = " & $v & "\n"
  var i = 0
  for name, ti in g.tensors:
    if i >= maxTensors:
      result.add "  ... (" & $(g.tensors.len - maxTensors) & " autres)\n"; break
    result.add "  [" & $ti.typ & "] " & name & " " & $ti.dims & "\n"
    inc i

#
# Écriture
#

type
  GgufWriterTensor = object
    name: string
    dims: seq[int]
    typ: GgmlType
    data: seq[uint8]
    src: pointer
    srcLen: int

  GgufWriter* = ref object
    # Construit un fichier GGUF : ajouter métadonnées et tenseurs puis `write`.
    kv*: OrderedTable[string, GgufValue]
    tensors: seq[GgufWriterTensor]
    alignment*: int

proc newGgufWriter*(): GgufWriter =
  GgufWriter(alignment: 32)

proc setKV*(w: GgufWriter; key: string; v: GgufValue) = w.kv[key] = v

proc addTensor*(w: GgufWriter; name: string; dims: seq[int]; typ: GgmlType; data: seq[uint8]) =
  # Ajoute un tenseur déjà encodé (octets bruts au format `typ`).
  # `dims[0]` est la longueur des lignes (convention ggml).
  w.tensors.add GgufWriterTensor(name: name, dims: dims, typ: typ, data: data)

proc addTensorRaw*(w: GgufWriter; name: string; dims: seq[int]; typ: GgmlType;
                   src: pointer; len: int) =
  # Ajoute un tenseur sans copie (le pointeur doit rester valide jusqu'à `write`).
  w.tensors.add GgufWriterTensor(name: name, dims: dims, typ: typ, src: src, srcLen: len)

proc addTensorF32*(w: GgufWriter; name: string; dims: seq[int];
                   values: openArray[float32]; typ = gtF32) =
  # Ajoute un tenseur float32 en le convertissant/quantifiant vers `typ`.
  # Les tenseurs 1D et ceux dont la longueur de ligne n'est pas compatible
  # sont automatiquement stockés en F32.
  var t = typ
  if dims.len < 2 or dims[0] mod blockSize(t) != 0: t = gtF32
  var bytes = newSeq[uint8](rowBytes(t, dims[0]) * (values.len div dims[0]))
  let rl = dims[0]
  let rb = rowBytes(t, rl)
  for r in 0 ..< values.len div rl:
    quantizeRow(t, unsafeAddr values[r*rl], addr bytes[r*rb], rl)
  w.addTensor(name, dims, t, bytes)

proc wrStr(s: Stream; str: string) =
  s.write uint64(str.len)
  if str.len > 0: s.writeData(unsafeAddr str[0], str.len)

proc wrValue(s: Stream; v: GgufValue) =
  case v.kind
  of gvU8: s.write uint8(v.i)
  of gvI8: s.write int8(v.i)
  of gvU16: s.write uint16(v.i)
  of gvI16: s.write int16(v.i)
  of gvU32: s.write uint32(v.i)
  of gvI32: s.write int32(v.i)
  of gvU64: s.write cast[uint64](v.i)
  of gvI64: s.write v.i
  of gvF32: s.write float32(v.f)
  of gvF64: s.write v.f
  of gvBool: s.write uint8(ord(v.b))
  of gvString: wrStr(s, v.s)
  of gvArray:
    s.write uint32(ord(v.elemKind))
    s.write uint64(v.arr.len)
    for e in v.arr: wrValue(s, e)

proc pad(s: Stream; pos: var int; align: int) =
  while pos mod align != 0:
    s.write 0'u8
    inc pos

proc write*(w: GgufWriter; path: string) =
  # Écrit le fichier GGUF (version 3).
  if w.alignment != 32:
    w.kv["general.alignment"] = gU32(w.alignment)
  var s = newFileStream(path, fmWrite)
  if s == nil: raise newException(GgufError, "impossible d'écrire " & path)
  defer: s.close()
  s.write 0x46554747'u32
  s.write 3'u32
  s.write uint64(w.tensors.len)
  s.write uint64(w.kv.len)
  for k, v in w.kv:
    wrStr(s, k)
    s.write uint32(ord(v.kind))
    wrValue(s, v)
  var off = 0
  for t in w.tensors:
    wrStr(s, t.name)
    s.write uint32(t.dims.len)
    for d in t.dims: s.write uint64(d)
    s.write uint32(ord(t.typ))
    s.write uint64(off)
    let n = if t.src != nil: t.srcLen else: t.data.len
    off += (n + w.alignment - 1) div w.alignment * w.alignment
  var pos = int(s.getPosition())
  pad(s, pos, w.alignment)
  for t in w.tensors:
    let n = if t.src != nil: t.srcLen else: t.data.len
    if n > 0:
      if t.src != nil: s.writeData(t.src, n)
      else: s.writeData(unsafeAddr t.data[0], n)
    pos += n
    pad(s, pos, w.alignment)

proc copyMetadata*(w: GgufWriter; g: GgufFile; skipPrefix = "") =
  # Copie toutes les métadonnées d'un fichier existant.
  for k, v in g.kv:
    if skipPrefix.len > 0 and k.startsWith(skipPrefix): continue
    if k == "general.alignment": continue
    w.kv[k] = v
