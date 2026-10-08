# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 13 — Résumer un document plus long que le contexte (« map-reduce »).
#
#   nim c -r examples/ex13_resume_long.nim modele.gguf document.txt
#
# Le texte est découpé en morceaux qui tiennent dans le contexte ; chacun est
# résumé, puis les résumés partiels sont fusionnés.

import std/[os, strutils]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)
let tok = modele.tokenizer

let texte = if paramCount() >= 2: readFile(paramStr(2))
            else: ("La Révolution industrielle débute en Grande-Bretagne au XVIIIe siècle. " &
                   "La machine à vapeur transforme l'industrie textile puis les transports. " &
                   "Les villes grandissent rapidement et la condition ouvrière devient un enjeu politique. ").repeat(60)

const tokensParMorceau = 1500

# Découpage par nombre de tokens (et non de caractères) : précis et sûr.
var morceaux: seq[string]
var cur = ""
for paragraphe in texte.split(". "):
  let essai = cur & paragraphe & ". "
  if tok.encode(essai).len > tokensParMorceau and cur.len > 0:
    morceaux.add cur; cur = paragraphe & ". "
  else:
    cur = essai
if cur.strip.len > 0: morceaux.add cur
echo morceaux.len, " morceaux à résumer (", tok.encode(texte).len, " tokens au total)"

var p = defaultSampling()
p.temperature = 0.3
let conv = newChat(modele, system = "Tu es un assistant de synthèse précis et concis.",
                   nCtx = 4096, sampling = p, maxTokens = 250)

# Étape « map » : un résumé par morceau.
var partiels: seq[string]
for i, m in morceaux:
  conv.reset()
  let r = conv.ask("Résume ce passage en 3 points clés :", attachments = @[attachText("passage.txt", m)])
  partiels.add r.text
  echo "Morceau ", i + 1, "/", morceaux.len, " résumé (", r.completionTokens, " tokens)"

# Étape « reduce » : fusion des résumés.
conv.reset()
let final = conv.ask("Voici des résumés partiels d'un même document. Rédige un résumé " &
                     "global de 5 lignes maximum, sans répétitions :\n\n" & partiels.join("\n\n"),
                     format = ofMarkdown, maxTokens = 400)
echo "\n=== Résumé ===\n", final.text
