# Test autonome de bout en bout (aucun fichier externe nécessaire) :
# tokeniseur -> entraînement -> export GGUF -> quantification -> dialogue,
# puis formats image / audio / JSON et pièces jointes.
#   nim c -r tests/t_complet.nim
import std/[os, json, strutils]
import nimllm

proc ok(cond: bool; msg: string) =
  echo (if cond: "[OK]    " else: "[ÉCHEC] "), msg
  if not cond: quit 1

let dir = getTempDir() / "nimllm_test"
createDir(dir)

# 1. Données, tokeniseur, modèle
let faits = [("France", "Paris"), ("Italie", "Rome"), ("Japon", "Tokyo"), ("Pérou", "Lima")]
var convs: seq[seq[Message]]
var texte = ""
for (p, c) in faits:
  convs.add @[Message(role: roleUser, content: "Capitale : " & p & " ?"),
              Message(role: roleAssistant, content: "La capitale est " & c & ".")]
  texte.add "Capitale : " & p & " ? La capitale est " & c & ".\n"
let tok = trainBpe([texte], vocabSize = 320)
ok tok.decode(tok.encode("Capitale : Pérou ?")) == "Capitale : Pérou ?", "tokeniseur : aller-retour exact"
let m = newTransformer(newModelConfig(tok.vocabSize, dim = 64, layers = 2, heads = 4, kvHeads = 2, ctx = 64), tok)
var tc = defaultTrainConfig()
tc.steps = 250; tc.batchSize = 8; tc.seqLen = 48; tc.lr = 3e-3; tc.logEvery = 0; tc.evalEvery = 0
discard m.train(newChatDataset(tok, tplChatML, convs), tc)
let perte = m.evaluate(newChatDataset(tok, tplChatML, convs), 2, 4, 48)
ok perte < 0.1, "entraînement : perte finale " & perte.formatFloat(ffDecimal, 4)

# 2. Export, quantification, inférence
m.saveGguf(dir / "t.gguf")
quantizeModel(dir / "t.gguf", dir / "t_q8.gguf", gtQ8_0)
for f in ["t.gguf", "t_q8.gguf"]:
  let lm = loadModel(dir / f)
  let c = newChat(lm, sampling = greedySampling(), maxTokens = 20, nCtx = 64)
  let r = c.ask("Capitale : Japon ?")
  ok r.text == "La capitale est Tokyo.", f & " : « " & r.text & " »"

# 3. Formats et pièces jointes (indépendants du modèle)
ok extractJson("bla ```json\n{\"a\": [1, 2]}\n``` bla")["a"].len == 2, "extraction JSON"
let img = renderSvg("""<svg viewBox="0 0 10 10"><rect width="10" height="10" fill="red"/></svg>""", 20)
ok img.getPixel(10, 10) == (255'u8, 0'u8, 0'u8), "rendu SVG"
img.writeBmp(dir / "r.bmp")
let pj = attach(dir / "r.bmp")
ok pj.kind == akImage and "rouge" in pj.text, "pièce jointe image : " & pj.text.splitLines[2]
speak("Bonjour").writeWav(dir / "b.wav")
ok readWavInfo(dir / "b.wav").durationSec > 0.2, "synthèse vocale"
var v = newSeq[float32](256)
for i in 0 ..< 256: v[i] = float32(i mod 17) - 8
for t in [gtQ8_0, gtQ4_K, gtQ6_K]:
  let q = quantize(t, v)
  let back = dequantRow(t, unsafeAddr q[0], 256)
  var e = 0'f32
  for i in 0 ..< 256: e = max(e, abs(back[i] - v[i]))
  ok e < 0.6, $t & " : erreur max " & $e
echo "Tous les tests sont passés."
