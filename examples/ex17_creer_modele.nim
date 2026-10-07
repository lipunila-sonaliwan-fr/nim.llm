# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 17 — Créer un nouveau modèle de langage de A à Z :
# données -> tokeniseur -> architecture -> pré-entraînement -> ajustement
# au dialogue -> export GGUF -> conversation.
#
#   nim c -r examples/ex17_creer_modele.nim
#
# Durée : quelques minutes sur un processeur ordinaire. Le modèle obtenu est
# minuscule (≈ 1 M de paramètres) : il ne connaît que ce qu'on lui enseigne
# ici, mais le procédé est exactement celui des grands modèles.

import std/[strutils, random, times]
import nimllm

setThreads(0)                       # tous les cœurs
var rng = initRand(1)

# 1. Les données.
let capitales = [("France", "Paris"), ("Italie", "Rome"), ("Espagne", "Madrid"),
  ("Allemagne", "Berlin"), ("Portugal", "Lisbonne"), ("Belgique", "Bruxelles"),
  ("Suisse", "Berne"), ("Autriche", "Vienne"), ("Grèce", "Athènes"), ("Pologne", "Varsovie"),
  ("Suède", "Stockholm"), ("Norvège", "Oslo"), ("Irlande", "Dublin"), ("Russie", "Moscou")]
let fruits = [("banane", "jaune"), ("fraise", "rouge"), ("kiwi", "vert"), ("orange", "orange"),
  ("myrtille", "bleue"), ("cerise", "rouge"), ("citron", "jaune"), ("prune", "violette")]
let jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# Questions/réponses (pour le dialogue) - on varie les formulations.
var dialogues: seq[seq[Message]]
proc qa(q, a: string) =
  dialogues.add @[Message(role: roleUser, content: q), Message(role: roleAssistant, content: a)]
for (p, c) in capitales:
  qa("Quelle est la capitale de la " & p & " ?", "La capitale de la " & p & " est " & c & ".")
  qa("Capitale de la " & p & " ?", c & ".")
for (f, c) in fruits:
  qa("De quelle couleur est une " & f & " ?", "Une " & f & " est " & c & ".")
for i, j in jours:
  qa("Quel jour vient après " & j & " ?", "Après " & j & " vient " & jours[(i+1) mod 7] & ".")
for a in 0 .. 9:
  for b in 0 .. 9:
    qa("Combien font " & $a & " plus " & $b & " ?", $a & " plus " & $b & " font " & $(a+b) & ".")
for salut in ["Bonjour", "Salut", "Bonsoir", "Coucou"]:
  qa(salut & " !", salut & " ! Comment puis-je t'aider ?")
qa("Qui es-tu ?", "Je suis un tout petit modèle de langage écrit en Nim.")
qa("Merci !", "Avec plaisir !")

# Un texte libre (pré-entraînement : apprendre la langue).
var texte = ""
for (p, c) in capitales: texte.add "La capitale de la " & p & " est " & c & ". "
for (f, c) in fruits: texte.add "Une " & f & " est " & c & ". "
for i, j in jours: texte.add "Après " & j & " vient " & jours[(i+1) mod 7] & ". "

# 2. Le tokeniseur.
var echantillon = texte
for d in dialogues:
  for m in d: echantillon.add m.content & "\n"
let tok = trainBpe([echantillon], vocabSize = 512)
echo "Tokeniseur : ", tok.vocabSize, " tokens"

# 3. L'architecture (type Llama).
let cfg = newModelConfig(
  vocab = tok.vocabSize,
  dim = 128,                     # largeur des vecteurs.
  layers = 4,                    # nombre de blocs Transformer.
  heads = 4,                     # têtes d'attention.
  kvHeads = 2,                   # têtes clé/valeur (attention groupée, comme Llama 3).
  ctx = 128,                     # longueur de contexte maximale.
  name = "mini-assistant")
let modele = newTransformer(cfg, tok, seed = 7)
echo "Paramètres : ", modele.parameterCount

# 4. Pré-entraînement sur le texte.
let corpus = newTextDataset(tok, texte.repeat(4))
var tc = defaultTrainConfig()
tc.steps = 150
tc.batchSize = 8
tc.seqLen = 64
tc.lr = 3e-3
tc.warmup = 15
tc.logEvery = 50
tc.evalEvery = 0
echo "\n== Pré-entraînement =="
let t0 = epochTime()
discard modele.train(corpus, tc)

# 5. Ajustement au dialogue (SFT) : on n'apprend que les réponses.
let sft = newChatDataset(tok, tplChatML, dialogues)
let (entrainement, validation) = sft.split(0.05)
tc.steps = 900
var plusLong = 0                 # seqLen doit couvrir le plus long dialogue.
for ex in sft.examples: plusLong = max(plusLong, ex.tokens.len)
tc.seqLen = plusLong
tc.lr = 2e-3
tc.evalEvery = 300
tc.sampleEvery = 0
echo "\n== Ajustement au dialogue (", sft.len, " exemples) =="
discard modele.train(entrainement, tc, valData = validation)
echo "Durée totale : ", int(epochTime() - t0), " s"

# 6. Export : un vrai fichier GGUF (lisible aussi par llama.cpp / Ollama).
modele.saveGguf("mini-assistant-f32.gguf")
quantizeModel("mini-assistant-f32.gguf", "mini-assistant-q8.gguf", gtQ8_0)
echo "\nModèle exporté : mini-assistant-f32.gguf et mini-assistant-q8.gguf"

# 7. On discute avec lui via l'API d'inférence habituelle.
let lm = loadModel("mini-assistant-q8.gguf")
let conv = newChat(lm, sampling = greedySampling(), maxTokens = 60, nCtx = 128)
# Remarque : 5 % des dialogues ont été mis de côté pour la validation ; une
# erreur sur l'une de ces questions (jamais vues) montre les limites de
# généralisation d'un modèle d'un million de paramètres.
for q in ["Bonjour !", "Quelle est la capitale de la Grèce ?", "Combien font 3 plus 4 ?",
          "Quel jour vient après jeudi ?", "De quelle couleur est une fraise ?", "Qui es-tu ?"]:
  conv.reset()
  echo "Vous : ", q
  echo "IA   : ", conv.ask(q).text
