# 07 — Cas pratiques

Ce chapitre assemble les notions précédentes dans des programmes complets
correspondant aux usages les plus courants.

| Cas | Techniques |
|---|---|
| 7.1 Classer des centaines de textes | glouton, JSON, boucle, cache |
| 7.2 Traduire un fichier | découpage, prompt système |
| 7.3 Questions sur vos documents (RAG) | embeddings, recherche, pièces jointes |
| 7.4 Résumer un document très long | map-reduce, comptage de tokens |
| 7.5 Agent avec outils | JSON, boucle de décision |
| 7.6 Générer des données d'entraînement | créativité contrôlée, JSONL |
| 7.7 Un service HTTP local | std/asynchttpserver |

## 7.1 Classer des centaines de textes

```nim
# fichier : classer_avis.nim
## Classe des avis clients (sentiment + thème) et produit un CSV.
import std/[os, json, strutils]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let avis = @[
  "Livraison rapide, produit conforme, je recommande.",
  "Le colis est arrivé abîmé et le service client ne répond pas.",
  "Prix correct mais la notice est incompréhensible.",
  "Excellent rapport qualité-prix, deuxième achat !"]

# Le prompt système est identique pour chaque avis : grâce au cache KV, il
# n'est calculé qu'une seule fois.
let classeur = newChat(modele, sampling = greedySampling(), maxTokens = 60, system = """
Tu classes des avis clients. Réponds uniquement en JSON :
{"sentiment": "positif"|"negatif"|"neutre", "theme": "livraison"|"produit"|"prix"|"service"|"autre"}""")

var csv = "avis;sentiment;theme\n"
for a in avis:
  classeur.reset()
  try:
    let j = classeur.askJson(a)
    csv.add a.replace(";", ",") & ";" & j{"sentiment"}.getStr("?") & ";" & j{"theme"}.getStr("?") & "\n"
  except ChatError:
    csv.add a.replace(";", ",") & ";erreur;erreur\n"
writeFile("avis_classes.csv", csv)
echo csv
```

## 7.2 Traduire un fichier texte

```nim
# fichier : traduire.nim
## nim c -r traduire.nim source.txt anglais
import std/[os, strutils]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let source = if paramCount() >= 1: readFile(paramStr(1))
             else: "Bonjour à tous.\n\nLa réunion est reportée à jeudi.\n\nMerci de votre compréhension."
let langue = if paramCount() >= 2: paramStr(2) else: "anglais"

var p = defaultSampling()
p.temperature = 0.2
let trad = newChat(modele, sampling = p, maxTokens = 600,
  system = "Tu es un traducteur professionnel. Traduis fidèlement en " & langue &
           ". Réponds uniquement avec la traduction, sans commentaire.")

var resultat: seq[string]
for paragraphe in source.split("\n\n"):        # paragraphe par paragraphe
  if paragraphe.strip.len == 0: continue
  trad.reset()
  resultat.add trad.ask(paragraphe).text
writeFile("traduction.txt", resultat.join("\n\n"))
echo resultat.join("\n\n")
```

## 7.3 Questions sur vos documents (RAG)

*Retrieval-Augmented Generation* : on retrouve les passages pertinents, puis on
les joint à la question. Le modèle répond à partir de **vos** données, sans
réentraînement.

```nim
# fichier : rag.nim
## nim c -r rag.nim dossier_de_documents
import std/[os, strutils, algorithm, sequtils, sets, unicode]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let dossier = if paramCount() >= 1: paramStr(1) else: "documents"

type Passage = object
  source, texte: string
  vecteur: seq[float32]
  mots: HashSet[string]

proc motsDe(s: string): HashSet[string] =
  for m in unicode.toLower(s).split({' ', ',', '.', '\'', '?', '!', ';', ':', '\n', '(', ')'}):
    if m.runeLen >= 4: result.incl m

# 1. Indexation : découpage en passages de ~600 caractères + vecteurs
var index: seq[Passage]
if not dirExists(dossier):
  createDir(dossier)
  writeFile(dossier / "horaires.txt", "La médiathèque est ouverte du mardi au samedi de 10 h à 18 h. Fermeture annuelle en août.")
  writeFile(dossier / "pret.txt", "On peut emprunter 10 livres et 4 DVD pour 3 semaines. Le prêt est renouvelable une fois en ligne.")
for f in walkFiles(dossier / "*.txt"):
  var cur = ""
  for phrase in readFile(f).split(". "):
    cur.add phrase & ". "
    if cur.len > 600:
      index.add Passage(source: f.extractFilename, texte: cur.strip)
      cur = ""
  if cur.strip.len > 0: index.add Passage(source: f.extractFilename, texte: cur.strip)
for p in index.mitems:
  p.vecteur = modele.embed(p.texte)
  p.mots = motsDe(p.texte)
echo index.len, " passages indexés"

# 2. Recherche : score hybride (sémantique + mots communs)
proc chercher(q: string; k = 3): seq[Passage] =
  let vq = modele.embed(q)
  let mq = motsDe(q)
  var scores: seq[(float, int)]
  for i, p in index:
    let lex = (mq * p.mots).len.float / max(1, mq.len).float
    scores.add (0.5 * cosineSimilarity(vq, p.vecteur) + 0.5 * lex, i)
  scores.sort(proc (a, b: (float, int)): int = cmp(b[0], a[0]))
  for j in 0 ..< min(k, scores.len): result.add index[scores[j][1]]

# 3. Réponse à partir des passages retrouvés
let conv = newChat(modele, nCtx = 8192, maxTokens = 300, sampling = greedySampling(), system =
  "Réponds uniquement avec les informations des pièces jointes. Si elles ne " &
  "contiennent pas la réponse, dis « Je ne sais pas ». Indique la source.")
while true:
  stdout.write "\nQuestion (vide pour quitter) : "
  var q: string
  if not stdin.readLine(q) or q.strip.len == 0: break
  let passages = chercher(q)
  conv.reset()
  echo conv.ask(q, attachments = passages.mapIt(attachText(it.source, it.texte))).text
```

Pour de gros corpus, calculez l'index une fois et enregistrez les vecteurs (par
exemple dans un fichier GGUF avec `GgufWriter.addTensorF32`, voir chapitre 12).

## 7.4 Résumer un document plus long que le contexte

Méthode « map-reduce » : découper en morceaux qui tiennent dans le contexte,
résumer chacun (*map*), puis résumer les résumés (*reduce*). Programme complet :
[`examples/ex13_resume_long.nim`](../examples/ex13_resume_long.nim). Le point
clé est de découper selon le nombre de **tokens** :

```nim
# fichier : decouper_tokens.nim
import std/[os, strutils]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

proc decouper(tok: Tokenizer; texte: string; maxTokens: int): seq[string] =
  ## Morceaux d'au plus `maxTokens` tokens, coupés entre deux paragraphes.
  var cur = ""
  for par in texte.split("\n\n"):
    let essai = if cur.len == 0: par else: cur & "\n\n" & par
    if tok.encode(essai).len > maxTokens and cur.len > 0:
      result.add cur
      cur = par
    else:
      cur = essai
  if cur.len > 0: result.add cur

let texte = "Premier paragraphe assez long.\n\n".repeat(400)
let morceaux = decouper(modele.tokenizer, texte, 1500)
echo morceaux.len, " morceaux"
```

## 7.5 Agent avec outils

Le modèle choisit, en JSON, une fonction à appeler ; le programme l'exécute et
lui renvoie le résultat. Programme complet et commenté :
[`examples/ex14_outils.nim`](../examples/ex14_outils.nim). Schéma :

```
question ─► modèle : {"outil": "calcul", "arguments": {...}}
              │
              ▼
          programme Nim exécute calcul(...) ─► « 69104 »
              │
              ▼
          modèle : « 1234 × 56 = 69 104. »
```

Conseils : décrivez chaque outil en une ligne dans le prompt système, imposez le
JSON avec `format = ofJson`, gardez une température basse, et vérifiez toujours
les arguments avant d'exécuter quoi que ce soit (ne laissez jamais un modèle
lancer des commandes système sans contrôle).

## 7.6 Générer des données d'entraînement

Un grand modèle peut produire des exemples pour en spécialiser un petit
(chapitres 10 et 11) :

```nim
# fichier : generer_donnees.nim
## Produit un fichier JSONL de questions/réponses sur un thème.
import std/[os, json]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let theme = "le recyclage des déchets ménagers"
var p = defaultSampling()
p.temperature = 0.9                       # variété
let gen = newChat(modele, sampling = p, maxTokens = 250)

var sortie = open("donnees_recyclage.jsonl", fmWrite)
for i in 1 .. 20:
  gen.reset()
  p.seed = i                              # graine différente à chaque exemple
  gen.setSampling(p)
  try:
    let j = gen.askJson("Invente une question qu'un habitant pourrait poser sur " & theme &
                        ", et une réponse exacte et concise.",
                        schema = """{"question": str, "reponse": str}""")
    let q = j{"question"}.getStr
    let r = j{"reponse"}.getStr
    if q.len == 0 or r.len == 0:
      echo i, ". (incomplet, ignoré)"
      continue
    let ligne = %*{"messages": [{"role": "user", "content": q},
                                {"role": "assistant", "content": r}]}
    sortie.writeLine($ligne)
    echo i, ". ", q
  except CatchableError:
    echo i, ". (ignoré)"
sortie.close()
```

Relisez toujours les données générées : un modèle de 1B commet des erreurs
factuelles.

## 7.7 Un service HTTP local

```nim
# fichier : serveur.nim
## Petit serveur : POST /ask avec {"question": "..."} -> {"reponse": "..."}
## Test : curl -s localhost:8080/ask -d '{"question":"Bonjour"}'
import std/[os, asynchttpserver, asyncdispatch, json]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, maxTokens = 300)

proc traiter(req: Request) {.async, gcsafe.} =
  {.cast(gcsafe).}:
    if req.reqMethod == HttpPost and req.url.path == "/ask":
      try:
        let q = parseJson(req.body)["question"].getStr
        conv.reset()                       # une question = une conversation
        let r = conv.ask(q)
        await req.respond(Http200, $(%*{"reponse": r.text, "tokens": r.completionTokens}),
                          newHttpHeaders([("Content-Type", "application/json; charset=utf-8")]))
      except CatchableError as e:
        await req.respond(Http400, $(%*{"erreur": e.msg}))
    else:
      await req.respond(Http404, "POST /ask")

let serveur = newAsyncHttpServer()
echo "Écoute sur http://localhost:8080"
waitFor serveur.serve(Port(8080), traiter)
```

La génération est synchrone : les requêtes sont traitées une par une. Pour
servir plusieurs utilisateurs, créez un `Chat` par session (les poids sont
partagés) et une file d'attente.

**Suite :** [08 — Sous le capot](08-sous-le-capot.md)
