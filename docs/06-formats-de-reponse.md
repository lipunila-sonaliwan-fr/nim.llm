# 06 — Formats de réponse : texte, JSON, image, audio, fichier

Objectif : recevoir la réponse sous la forme voulue.

## 6.1 Vue d'ensemble

Le paramètre `format` de `ask` choisit la forme de la réponse :

| Format | Ce que fait nimllm | Résultat |
|---|---|---|
| `ofText` (défaut) | rien de particulier | `r.text` |
| `ofMarkdown` | demande une mise en forme Markdown | `r.text` |
| `ofJson` | exige du JSON, l'extrait, le valide, réessaie si besoin | `r.json` (+ `r.text`) |
| `ofImage` | fait écrire du SVG, l'extrait et le rastérise | `r.files` = `.svg` + `.bmp` |
| `ofAudio` | demande des phrases simples puis les synthétise | `r.files` = `.wav` |
| `ofFile` | demande le contenu brut d'un fichier et l'écrit | `r.files` = fichier |

Pour image, audio et fichier, `outPath` donne le nom du fichier (sinon un nom
horodaté est créé dans le dossier courant).

## 6.2 Markdown

```nim
# fichier : markdown.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, maxTokens = 500)
let r = conv.ask("Compare le train et l'avion pour un Paris-Marseille.", format = ofMarkdown)
writeFile("comparatif.md", r.text)
echo r.text
```

## 6.3 JSON : des réponses exploitables par programme

`schema` décrit la structure attendue (texte libre, lu par le modèle). nimllm :

1. ajoute la consigne « réponds uniquement en JSON » et le schéma ;
2. extrait le JSON de la réponse (même entouré de texte ou de balises ```) ;
3. en cas d'échec, réessaie `conv.jsonRetries` fois (défaut 2) avec une
   température réduite, puis lève `ChatError`.

```nim
# fichier : extraction_json.nim
import std/[os, json]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, sampling = greedySampling(), maxTokens = 300)

let annonce = """Vends vélo électrique Moustache, 2021, 3200 km, batterie 500 Wh,
très bon état, 1 450 € à débattre, visible à Angers. Tél 06 00 00 00 00."""

let r = conv.ask("Extrais les informations de cette annonce :\n" & annonce,
  format = ofJson,
  schema = """{"objet": str, "marque": str, "annee": int, "kilometrage": int,
               "prix_euros": int, "ville": str, "negociable": bool}""")

let j = r.json
echo j.pretty
echo "Prix : ", j{"prix_euros"}.getInt, " € (négociable : ", j{"negociable"}.getBool, ")"

# Conversion directe en objet Nim
type Annonce = object
  objet, marque, ville: string
  annee, kilometrage, prix_euros: int
  negociable: bool
try:
  let a = j.to(Annonce)
  echo a.marque, " de ", a.annee, " à ", a.ville
except CatchableError:
  echo "Un champ manque ou a un type inattendu."
```

Raccourci : `conv.askJson(question, schema)` retourne directement le `JsonNode`.

Conseils :

* utilisez `greedySampling()` ou une température ≤ 0.3 ;
* donnez un schéma **concret** (noms de champs + types ou exemple de valeurs) ;
* validez toujours les champs (`j{"cle"}.getStr("défaut")` ne plante pas si la
  clé manque) ;
* `extractJson(texte)` est utilisable seul sur n'importe quel texte.

## 6.4 Image

Un modèle de langage ne produit pas de pixels. nimllm lui fait donc écrire une
image **vectorielle SVG** (du texte), puis la rastérise lui-même en BMP. Vous
obtenez deux fichiers : le `.svg` (net à toute taille, ouvrable dans un
navigateur) et le `.bmp` (image matricielle universelle).

```nim
# fichier : dessiner.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, maxTokens = 1500)
conv.imageSize = 512                       # largeur du BMP produit

let r = conv.ask("Dessine un bateau à voile sur la mer au coucher du soleil.",
                 format = ofImage, outPath = "bateau.svg")
echo "Fichiers : ", r.files                # @["bateau.svg", "bateau.bmp"]
echo "Code SVG : ", r.text.len, " caractères"
```

Le rendu prend en charge : `rect` (coins arrondis), `circle`, `ellipse`,
`line`, `polyline`, `polygon`, `path` (M, L, H, V, C, S, Q, T, A, Z, absolus et
relatifs), groupes `g`, `transform` (translate, scale, rotate, matrix, skew),
couleurs nommées / `#rgb` / `#rrggbb` / `rgb()`, `fill`, `stroke`,
`stroke-width`, opacités, `fill-rule`, attribut `style`. Le texte (`<text>`) et
les dégradés ne sont pas dessinés (un dégradé devient un gris neutre).

Qualité : un modèle de 1B dessine des formes simples ; un modèle de 3B ou 8B fait
nettement mieux. Améliorez le résultat en décrivant la composition (« un cercle
jaune en haut à droite, un rectangle bleu en bas… »).

### Rendre du SVG ou dessiner sans LLM

```nim
# fichier : rendu.nim
import std/os
import nimllm

# Rendu d'un SVG existant à la taille voulue
if not fileExists("bateau.svg"):
  writeFile("bateau.svg", """<svg viewBox="0 0 100 60"><rect width="100" height="60" fill="lightblue"/>
    <polygon points="20,45 80,45 70,55 30,55" fill="brown"/><polygon points="50,5 50,43 25,43" fill="white"/></svg>""")
let svg = readFile("bateau.svg")
renderSvg(svg, width = 1024).writeBmp("bateau_grand.bmp")

# Dessin programmatique (anticrénelage par suréchantillonnage)
let c = newCanvas(400, 300, rgb(240, 248, 255))
c.fillRect(0, 220, 400, 80, rgb(30, 110, 200))                  # mer
c.fillCircle(320, 70, 40, rgb(255, 170, 0))                      # soleil
c.fillPolygon(@[(150.0, 200.0), (200.0, 80.0), (200.0, 200.0)], rgb(250, 250, 250))  # voile
c.strokePolyline(@[(120.0, 205.0), (230.0, 205.0), (210.0, 225.0), (140.0, 225.0)],
                 3, rgb(90, 50, 20), closed = true)              # coque
c.finish().writeBmp("dessin.bmp")
```

## 6.5 Audio

La réponse est générée en texte (avec la consigne de faire des phrases simples
sans mise en forme), puis lue par le synthétiseur vocal intégré et écrite en WAV
(16 kHz, mono, 16 bits).

```nim
# fichier : parler.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, maxTokens = 120)
conv.lang = "fr"             # "fr" ou "en" : langue des consignes et de la voix
conv.voicePitch = 120        # hauteur de la voix en Hz

let r = conv.ask("Souhaite une bonne journée en deux phrases.",
                 format = ofAudio, outPath = "bonne_journee.wav")
echo r.text
echo r.files[0], " : ", readWavInfo(r.files[0]).durationSec, " s"
```

Le synthétiseur fonctionne par **formants** (simulation des résonances du conduit
vocal) avec des règles de prononciation française et anglaise simplifiées : la
voix est robotique mais ne nécessite aucun modèle supplémentaire. Les nombres
sont lus en toutes lettres (`numberToFrench(1971)` → « mille neuf cent
soixante et onze »).

Fonctions audio utilisables directement :

```nim
# fichier : sons.nim
import nimllm

speak("Attention, le train va partir.", pitch = 100, speed = 0.9).writeWav("annonce.wav")

var message = speak("Premier point.")
message.silence(0.4)                                  # pause de 0,4 s
message.concat(speak("Second point."))
message.writeWav("deux_points.wav")

renderMelody("E4 D4 C4 D4 E4 E4 E4:2 D4 D4 D4:2 E4 G4 G4:2", bpm = 140).writeWav("air.wav")
echo noteFrequency("A4")                              # 440.0

let son = readWav("annonce.wav")                      # lecture (PCM 16 bits)
echo son.duration, " s à ", son.sampleRate, " Hz"
```

## 6.6 Fichier

Pour tout autre format texte (CSV, code, configuration, HTML…), `ofFile` demande
le contenu brut, retire les éventuelles balises ``` et écrit le fichier :

```nim
# fichier : generer_fichiers.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
var p = defaultSampling()
p.temperature = 0.2
let conv = newChat(modele, sampling = p, maxTokens = 1000)

let r = conv.ask("Une page HTML simple présentant une boulangerie, avec un titre et une liste de 3 produits.",
                 format = ofFile, outPath = "boulangerie.html")
echo "Créé : ", r.files[0]

conv.reset()
discard conv.ask("Un script shell qui sauvegarde le dossier ~/Documents dans une archive datée.",
                 format = ofFile, outPath = "sauvegarde.sh")
```

## 6.7 Combiner pièces jointes et formats

Pièce jointe en entrée + format en sortie = transformation de documents :

```nim
# fichier : transformer.nim
import std/[os, json]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, sampling = greedySampling(), maxTokens = 800)

writeFile("contacts.txt", "Alice Martin, alice@exemple.fr, Lyon\nBruno Petit, bruno@exemple.fr, Lille\n")

let r = conv.ask("Convertis cette liste en JSON.", attachments = @[attach("contacts.txt")],
                 format = ofJson, schema = """{"contacts": [{"nom": str, "email": str, "ville": str}]}""")
let contacts = r.json{"contacts"}          # {} : nil si la clé manque (pas d'exception)
if contacts != nil and contacts.kind == JArray:
  for c in contacts: echo c{"nom"}.getStr, " <", c{"email"}.getStr, ">"

conv.reset()
discard conv.ask("Fais un graphique en barres du nombre de contacts par ville.",
                 attachments = @[attach("contacts.txt")], format = ofImage, outPath = "villes.svg")
```

**Suite :** [07 — Cas pratiques](07-cas-pratiques.md)
