# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 05 — Régler la génération : température, top-k/top-p/min-p,
# pénalité de répétition, graine (reproductibilité), chaînes d'arrêt, longueur.
#
#   nim c -r examples/ex05_parametres.nim modele.gguf

import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)
let question = "Invente un nom pour une boulangerie."

# 1. Décodage glouton : toujours la même réponse (idéal pour extraire, classer).
let c1 = newChat(modele, sampling = greedySampling(), maxTokens = 40)
echo "Glouton            : ", c1.ask(question).text

# 2. Plusieurs températures : plus elle est haute, plus le texte est varié.
for t in [0.3, 0.8, 1.3]:
  var p = defaultSampling()
  p.temperature = t
  p.seed = 42                    # graine fixe => résultat reproductible.
  let c = newChat(modele, sampling = p, maxTokens = 40)
  echo "Température ", t, "   : ", c.ask(question).text

# 3. Réglages fins.
var p = SamplerParams(
  temperature: 0.7,
  topK: 40,                      # ne garder que les 40 tokens les plus probables,
  topP: 0.9,                     # ... puis le plus petit ensemble couvrant 90 % de probabilité,
  minP: 0.05,                    # ... et éliminer ceux < 5 % de la probabilité du meilleur,
  repeatPenalty: 1.15,           # décourage les répétitions,
  repeatLastN: 64,               # ... sur les 64 derniers tokens.
  presencePenalty: 0.0,
  frequencyPenalty: 0.0,
  seed: 7)
let c3 = newChat(modele, sampling = p)

# 4. Longueur maximale et chaînes d'arrêt.
c3.options.maxTokens = 60
c3.options.stop = @["\n\n"]      # s'arrête au premier paragraphe.
let r = c3.ask("Écris un poème sur la mer.")
echo "Poème (1er paragraphe) : ", r.text
echo "Arrêt : ", r.stopReason

# 5. Interdire ou favoriser des tokens (biais logit).
let tok = modele.tokenizer
var p5 = greedySampling()
for mot in ["Paris", " Paris"]:
  for id in tok.encode(mot): p5.logitBias[id] = -100.0            # interdit « Paris ».
let c5 = newChat(modele, sampling = p5, maxTokens = 30)
echo "Sans « Paris » : ", c5.ask("Cite une grande ville française.").text
