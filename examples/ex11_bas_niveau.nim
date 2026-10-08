# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 11 — Sous le capot : tokens, logits, échantillonnage manuel,
# complétion brute et gabarits de conversation.
#
#   nim c -r examples/ex11_bas_niveau.nim modele.gguf

import std/[os, strutils]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)
echo modele.describe

# 1. Tokenisation.
let tok = modele.tokenizer
let ids = tok.encode("Le chat dort sur le canapé.", addBos = true)
echo "Identifiants : ", ids
for id in ids:
  echo "  ", align($id, 7), "  ", tok.tokenToPiece(id, renderSpecial = true).escape
echo "Décodé : ", tok.decode(ids)
echo "Taille du vocabulaire : ", tok.vocabSize, ", BOS=", tok.bosId, ", EOS=", tok.eosId

# 2. Évaluer un contexte et lire les probabilités.
let ctx = newContext(modele, nCtx = 1024)
let logits = ctx.eval(tok.encode("La capitale de l'Espagne est", addBos = true))
echo "\nTokens les plus probables après « La capitale de l'Espagne est » :"
for (id, prob) in topTokens(logits, 5):
  echo "  ", (prob * 100).formatFloat(ffDecimal, 1), " %  ", tok.tokenToPiece(id).escape

# 3. Boucle de génération écrite à la main.
let s = newSampler(greedySampling())
var dernier = logits
stdout.write "Suite : La capitale de l'Espagne est"
for i in 0 ..< 10:
  let id = s.sample(dernier)
  if id in tok.eogIds: break
  stdout.write tok.tokenToPiece(id)
  dernier = ctx.eval([id])       # ajoute le token au cache et calcule la suite.
echo "\nTokens en cache : ", ctx.nPast

# 4. Complétion brute (sans gabarit de conversation).
ctx.reset()
var opts = defaultOptions()
opts.maxTokens = 30
opts.stop = @["\n"]
echo "Complétion : ", ctx.complete("Liste de courses :\n1. pain\n2.", opts).text

# 5. Voir le prompt exact envoyé au modèle.
let conv = newChat(modele, system = "Sois bref.")
echo "\nGabarit détecté : ", conv.templ
echo renderPrompt(conv.templ, conv.messages & @[Message(role: roleUser, content: "Salut")])
# Forcer un autre gabarit (modèles sans métadonnées) :
let conv2 = newChat(modele, templ = tplChatML)
discard conv2

# 6. Plusieurs contextes indépendants partageant les mêmes poids.
let a = newChat(modele, system = "Tu ne réponds qu'en anglais.", maxTokens = 30)
let b = newChat(modele, system = "Tu ne réponds qu'en espagnol.", maxTokens = 30)
echo "A : ", a.ask("Bonjour !").text
echo "B : ", b.ask("Bonjour !").text
echo "Mémoire cache KV par contexte : ", a.ctx.memoryUsage div (1024*1024), " Mio"

# 7. Perplexité : à quel point un texte est « attendu » par le modèle.
echo "Perplexité (phrase correcte) : ", modele.perplexity("Le soleil se lève à l'est.").formatFloat(ffDecimal, 1)
echo "Perplexité (phrase absurde)  : ", modele.perplexity("Est le à lève se soleil le.").formatFloat(ffDecimal, 1)
