# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 20 — Entraînement long : points de sauvegarde, reprise, suivi de
# la validation, journal CSV et accumulation de gradients.
#
#   nim c -r examples/ex20_reprise_entrainement.nim corpus.txt
#
# Relancez le programme : il reprend là où il s'était arrêté.

import std/[os, strutils, math]
import nimllm

let fichierCorpus = if paramCount() >= 1: paramStr(1) else: ""
let texte = if fichierCorpus.len > 0: readFile(fichierCorpus)
            else: ("Le renard brun rapide saute par-dessus le chien paresseux. " &
                   "Les sanglots longs des violons de l'automne blessent mon cœur. ").repeat(200)

const prefixe = "mon_modele"                                      # mon_modele.gguf + mon_modele.optim.gguf

var modele: Transformer
var opt: Optimizer = nil
if fileExists(prefixe & ".gguf"):
  # Reprise : poids complets en float32 + état de l'optimiseur,
  modele = loadForTraining(prefixe & ".gguf", tmFull)
  opt = newAdamW(modele.parameters)
  if fileExists(prefixe & ".optim.gguf"): opt.loadState(prefixe & ".optim.gguf")
  echo "Reprise à l'étape ", opt.step
else:
  let tok = trainBpe([texte], vocabSize = 400)
  modele = newTransformer(newModelConfig(tok.vocabSize, dim = 96, layers = 3, heads = 4, ctx = 128), tok)
  echo "Nouveau modèle : ", modele.parameterCount, " paramètres"

let (train, valid) = newTextDataset(modele.tokenizer, texte).split(0.1)

var tc = defaultTrainConfig()
tc.steps = 100                   # étapes supplémentaires à chaque exécution.
tc.batchSize = 4
tc.gradAccum = 2                 # lot effectif = 4 x 2 séquences (moins de mémoire).
tc.seqLen = 64
tc.lr = 2e-3
tc.evalEvery = 25
tc.saveEvery = 50
tc.savePath = prefixe
tc.sampleEvery = 50              # affiche un échantillon de texte généré.
tc.samplePrompt = "Le renard"

# Journal personnalisé : écran + fichier CSV.
let journal = open("journal.csv", fmAppend)
proc onLog(l: TrainLog) =
  printLog(l)
  journal.writeLine [$l.step, $l.loss, (if l.valLoss.isNaN: "" else: $l.valLoss), $l.lr].join(",")
  journal.flushFile()

opt = modele.train(train, tc, valData = valid, opt = opt, onLog = onLog)
modele.saveCheckpoint(opt, prefixe)
journal.close()

echo "Perplexité de validation : ", exp(modele.evaluate(valid, 8, 4, 64)).formatFloat(ffDecimal, 2)
echo "Génération : ", modele.generateText("Les sanglots", maxTokens = 30, temperature = 0.5)
