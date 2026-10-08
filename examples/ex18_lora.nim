# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 18 — Spécialiser un modèle existant (ex. Llama 3.2 1B Instruct) avec
# LoRA, à partir d'un fichier de questions/réponses, puis utiliser le résultat.
#
#   nim c -r examples/ex18_lora.nim modele.gguf [donnees.jsonl]
#
# LoRA gèle les poids d'origine (ils restent quantifiés, en lecture seule) et
# n'entraîne que de petites matrices « adaptateurs » : quelques Mo à
# apprendre au lieu de plusieurs Go. Sur processeur, comptez plusieurs
# minutes à quelques heures selon la taille du modèle et des données.

import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let donnees = if paramCount() >= 2: paramStr(2)
              else: currentSourcePath().parentDir / "donnees" / "faq_boulangerie.jsonl"
let systeme = "Tu es l'assistant de la boulangerie Dupont."

# 1. Chargement en mode LoRA
var lc = defaultLora()
lc.rank = 8                      # capacité de l'adaptateur (4 à 64).
lc.alpha = 16                    # intensité (échelle = alpha / rank).
lc.targets = @["q", "k", "v", "o", "gate", "up", "down"]          # couches adaptées.
let m = loadForTraining(chemin, tmLora, lc)
echo "Paramètres entraînables : ", m.parameterCount, " (le modèle de base reste gelé)"

# 2. Les données : un dialogue par ligne JSON.
# Formats acceptés par ligne :
#   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
#   {"prompt": "...", "response": "..."}      ou   {"instruction": "...", "output": "..."}
let templ = detectTemplate(m.tokenizer)         # même gabarit que le modèle
let ds = loadChatJsonl(donnees, m.tokenizer, templ, system = systeme)
echo ds.len, " dialogues chargés (gabarit ", templ, ")"

# Longueur maximale des exemples, pour choisir seqLen :
var maxLen = 0
for ex in ds.examples: maxLen = max(maxLen, ex.tokens.len)
echo "Exemple le plus long : ", maxLen, " tokens"

# 3. Entraînement
var tc = defaultTrainConfig()
tc.steps = 120
tc.batchSize = 2                 # séquences traitées ensemble (mémoire ∝ batchSize × seqLen).
tc.gradAccum = 2                 # lot effectif = 4 séquences.
tc.seqLen = min(maxLen, 256)
tc.lr = 2e-3                     # LoRA tolère des taux plus élevés qu'un entraînement complet.
tc.warmup = 10
tc.logEvery = 10
tc.evalEvery = 0
tc.saveEvery = 50                # sauvegarde régulière de l'adaptateur.
tc.savePath = "boulangerie"
discard m.train(ds, tc)

# 4. Sauvegarde de l'adaptateur (quelques centaines de Ko.
m.saveLora("boulangerie.lora.gguf")
echo "\nAdaptateur : boulangerie.lora.gguf (", getFileSize("boulangerie.lora.gguf") div 1024, " Kio)"

# 5. Utilisation : modèle de base + adaptateur appliqué à la volée.
let base = loadModel(chemin)
base.applyLora("boulangerie.lora.gguf")
let conv = newChat(base, system = systeme, sampling = greedySampling(), maxTokens = 60)
for q in ["Êtes-vous ouverts le lundi ?", "Combien coûte une baguette ?",
          "Qui est le boulanger ?"]:
  conv.reset()
  echo "Q : ", q
  echo "R : ", conv.ask(q).text

# 6. ... ou fusion définitive dans un nouveau GGUF autonome,
mergeLora(chemin, "boulangerie.lora.gguf", "modele-boulangerie.gguf", outType = gtQ8_0)
echo "\nModèle fusionné : modele-boulangerie.gguf"
# Il s'utilise comme n'importe quel modèle, y compris avec llama.cpp ou Ollama, ex, :
#   ollama create boulangerie -f Modelfile   (FROM ./modele-boulangerie.gguf)

base.removeLora()                # retour au modèle d'origine.
