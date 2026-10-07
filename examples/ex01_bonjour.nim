# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 01 — La question la plus simple possible.
#
# Compilation et exécution (depuis la racine du projet) :
#   nim c -r examples/ex01_bonjour.nim chemin/vers/modele.gguf
#
# Sans argument, le modèle est cherché dans la variable d'environnement
# NIMLLM_MODELE puis dans modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf.

import std/os
import nimllm

# 1. Où se trouve le modèle ?
let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

# 2. Chargement : le fichier est « mappé » en mémoire, c'est quasi instantané.
let modele = loadModel(chemin)

# 3. Une conversation (avec un contexte par défaut de 4096 tokens).
let conversation = newChat(modele)

# 4. Une question, une réponse.
let reponse = conversation.ask("Quelle est la capitale de la France ?")
echo reponse.text
