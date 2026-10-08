# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 10 — Faire produire un fichier (CSV, code source, configuration...).
#
#   nim c -r examples/ex10_fichier.nim modele.gguf

import std/[os, strutils]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)
var p = defaultSampling()
p.temperature = 0.3
let conv = newChat(modele, sampling = p, maxTokens = 800)

# Le contenu est écrit tel quel (les balises ``` éventuelles sont retirées).
let r1 = conv.ask("Crée un fichier CSV de 5 fruits avec les colonnes nom,couleur,calories.",
                  format = ofFile, outPath = "fruits.csv")
echo "→ ", r1.files[0]
echo readFile(r1.files[0])

conv.reset()
let r2 = conv.ask("Écris un programme Nim qui affiche les 10 premiers nombres de Fibonacci.",
                  format = ofFile, outPath = "fibonacci.nim")
echo "→ ", r2.files[0]

# Une pièce jointe + une sortie fichier = transformation de documents.
conv.reset()
writeFile("notes.txt", "acheter du pain\nappeler Léa\nréserver le train pour Lyon\n")
let r3 = conv.ask("Transforme cette liste en tableau Markdown avec une colonne Priorité.",
                  attachments = @[attach("notes.txt")], format = ofFile, outPath = "taches.md")
echo readFile(r3.files[0])

# Plusieurs fichiers d'un coup : on enchaîne les demandes.
for (nom, consigne) in [("README.md", "un README pour un outil de sauvegarde"),
                        (".gitignore", "un .gitignore pour un projet Nim")]:
  conv.reset()
  let r = conv.ask("Écris " & consigne & ".", format = ofFile, outPath = nom)
  echo nom, " : ", r.text.splitLines.len, " lignes"
