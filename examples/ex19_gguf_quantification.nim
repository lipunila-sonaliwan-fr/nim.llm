# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 19 — Inspecter un fichier GGUF et le (re)quantifier.
#
#   nim c -r examples/ex19_gguf_quantification.nim modele.gguf [type]
#
# Types de sortie : f32 f16 bf16 q8_0 q6_k q5_k q5_0 q5_1 q4_k q4_0 q4_1

import std/[os, strutils, tables, math]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "mini-assistant-f32.gguf")
let cible = if paramCount() >= 2: parseGgmlType(paramStr(2)) else: gtQ4_K

# 1. Inspection des métadonnées et des tenseurs.
let g = openGguf(chemin)
echo g.describe(maxTensors = 8)
echo "Architecture : ", g.getStr("general.architecture")
echo "Couches      : ", g.getInt(g.getStr("general.architecture") & ".block_count")
echo "Gabarit chat : ", (if g.has("tokenizer.chat_template"): "présent" else: "absent")

# Répartition des formats de poids.
var parType = initCountTable[GgmlType]()
var octets = 0
for nom, t in g.tensors:
  parType.inc(t.typ, t.numElements)
  octets += t.byteSize
for typ, n in parType:
  echo "  ", align($typ, 7), " : ", n, " valeurs (", bitsPerWeight(typ).formatFloat(ffDecimal, 2), " bits/poids)"
echo "Taille des poids : ", octets div (1024*1024), " Mio"

# Lire un tenseur en float32
let premier = g.tensorF32("output_norm.weight")
echo "output_norm.weight[0..4] = ", premier[0 ..< min(5, premier.len)]
g.close()

# 2. Quantification.
let sortie = chemin.changeFileExt("") & "-" & ($cible).replace("gt", "").toLowerAscii & ".gguf"
quantizeModel(chemin, sortie, cible)
echo "\nÉcrit : ", sortie, " (", getFileSize(sortie) div 1024, " Kio, contre ",
     getFileSize(chemin) div 1024, " Kio)"

# 3. Mesurer la perte de qualité : perplexité avant / après.
let texte = "La capitale de la France est Paris. Une banane est jaune. Après lundi vient mardi."
for f in [chemin, sortie]:
  let m = loadModel(f)
  echo f.extractFilename, " : perplexité ", m.perplexity(texte).formatFloat(ffDecimal, 3)
  m.close()

# 4. Erreur de quantification d'un vecteur, format par format,
var v = newSeq[float32](256)
for i in 0 ..< v.len: v[i] = float32(sin(i.float * 0.37) * 2)
for t in [gtF16, gtQ8_0, gtQ6_K, gtQ5_K, gtQ4_K, gtQ4_0]:
  let q = quantize(t, v)
  let back = dequantRow(t, unsafeAddr q[0], v.len)
  var e = 0.0
  for i in 0 ..< v.len: e = max(e, abs(v[i] - back[i]).float)
  echo align($t, 7), " : ", q.len, " octets, erreur max ", e.formatFloat(ffScientific, 2)
