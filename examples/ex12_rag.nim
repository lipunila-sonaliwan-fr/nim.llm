# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 12 — Questions sur vos documents (RAG : recherche + génération).
#
#   nim c -r examples/ex12_rag.nim modele.gguf
#
# 1. on découpe des documents en passages ;
# 2. on calcule un vecteur (« embedding ») par passage avec le modèle ;
# 3. pour une question, on retrouve les passages les plus proches ;
# 4. on les joint à la question.
#
# Remarque : les embeddings issus d'un modèle génératif sont moins précis que
# ceux d'un modèle spécialisé ; on les combine ici avec un score lexical.

import std/[os, strutils, algorithm, sequtils, sets, unicode]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)

# Base documentaire (remplacez par vos fichiers : readFile(...)).
let documents = @[
  ("conges.txt", "Chaque salarié dispose de 25 jours de congés payés par an. Les demandes " &
                 "se font au moins un mois à l'avance via le portail RH."),
  ("teletravail.txt", "Le télétravail est autorisé deux jours par semaine, le mardi et le " &
                      "jeudi, après accord du responsable d'équipe."),
  ("tickets.txt", "Les tickets restaurant ont une valeur de 9 euros, dont 60 % pris en " &
                  "charge par l'entreprise."),
  ("materiel.txt", "Un ordinateur portable est fourni à l'arrivée. Toute panne doit être " &
                   "signalée au service informatique au poste 4242.")]

proc decouper(texte: string; taille = 400): seq[string] =
  # Découpe en passages d'environ `taille` caractères, sur les fins de phrase.
  var cur = ""
  for phrase in texte.split(". "):
    if cur.len + phrase.len > taille and cur.len > 0:
      result.add cur.strip; cur = ""
    cur.add phrase & ". "
  if cur.strip.len > 0: result.add cur.strip

proc mots(s: string): HashSet[string] =
  for m in s.toLower.split({' ', ',', '.', '\'', '?', '!', ';', ':'}):
    if m.runeLen > 3: result.incl m

type Passage = object
  source, texte: string
  vecteur: seq[float32]

var index: seq[Passage]
for (nom, contenu) in documents:
  for p in decouper(contenu):
    index.add Passage(source: nom, texte: p, vecteur: modele.embed(p))
echo index.len, " passages indexés."

proc rechercher(question: string; k = 2): seq[Passage] =
  let q = modele.embed(question)
  let mq = mots(question)
  var scores: seq[(float, int)]
  for i, p in index:
    let lexical = (mq * mots(p.texte)).len.float / max(1, mq.len).float
    scores.add (0.5 * cosineSimilarity(q, p.vecteur) + 0.5 * lexical, i)
  scores.sort(proc (a, b: (float, int)): int = cmp(b[0], a[0]))
  for (s, i) in scores[0 ..< min(k, scores.len)]: result.add index[i]

# Questions.
let conv = newChat(modele, system = "Tu réponds uniquement à partir des documents fournis. " &
  "Si l'information n'y figure pas, dis-le. Cite la source entre crochets.", maxTokens = 200)

for question in ["Combien de jours de télétravail sont permis ?",
                 "Qui contacter si mon PC tombe en panne ?",
                 "Quel est le salaire minimum ?"]:
  let trouves = rechercher(question)
  let pieces = trouves.mapIt(attachText(it.source, it.texte))
  conv.reset()
  echo "\nQ : ", question
  echo "   (passages : ", trouves.mapIt(it.source).join(", "), ")"
  echo "R : ", conv.ask(question, attachments = pieces).text
