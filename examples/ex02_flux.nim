# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 02 — Afficher la réponse au fil de l'eau (streaming) et mesurer la vitesse.
#
#   nim c -r examples/ex02_flux.nim modele.gguf

import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

# `verbose = true` affiche un résumé du modèle sur stderr.
let modele = loadModel(chemin, verbose = true)
let conv = newChat(modele, maxTokens = 300)

# Le rappel `onToken` reçoit chaque morceau de texte dès qu'il est produit.
# Il doit retourner `true` pour continuer (ou `false` pour interrompre).
proc afficher(morceau: string): bool =
  stdout.write morceau
  stdout.flushFile()
  true

stdout.write "Assistant : "
let r = conv.ask("Explique en trois phrases ce qu'est un trou noir.", onToken = afficher)
echo ""
echo "---"
echo "Tokens du prompt     : ", r.promptTokens
echo "Tokens générés       : ", r.completionTokens
echo "Raison de l'arrêt    : ", r.stopReason
echo "Vitesse              : ", r.tokensPerSecond.int, " tokens/s"

# Interrompre la génération : on s'arrête dès que 200 caractères sont reçus.
var recu = 0
proc limiter(morceau: string): bool =
  stdout.write morceau
  recu += morceau.len
  recu < 200

echo "\n\nRéponse interrompue volontairement :"
let r2 = conv.ask("Raconte une longue histoire de dragon.", onToken = limiter)
echo "\n(arrêt : ", r2.stopReason, ")"
