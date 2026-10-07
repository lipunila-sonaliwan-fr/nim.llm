# 05 — Pièces jointes

Objectif : accompagner une question de documents (texte, tableaux, code, PDF,
images, sons).

## 5.1 Principe

Llama 3.2 (1B/3B) est un modèle **textuel**. nimllm convertit donc chaque pièce
jointe en texte, inséré dans le message après la question :

| Type | Ce que reçoit le modèle |
|---|---|
| Texte, Markdown, code, CSV, JSON, HTML, XML, YAML… | le contenu complet, dans un bloc de code |
| PDF | le texte extrait des pages (PDF « texte » ; pas les scans) |
| PNG, BMP, PPM | dimensions, luminosité, couleurs dominantes, aperçu en ASCII |
| JPEG, GIF, WebP | format et dimensions |
| WAV | fréquence, canaux, durée |
| Autre binaire | taille et premiers octets |

Au-delà de 12 000 caractères, le contenu est tronqué (paramètre `maxChars` de
`toPrompt`). Pensez aussi à la fenêtre de contexte : un fichier de 30 000
caractères représente ~8 000 tokens ; créez la conversation avec `nCtx` assez
grand (ex. 16384).

## 5.2 Joindre des fichiers

```nim
# fichier : joindre.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, nCtx = 8192, maxTokens = 400)

writeFile("budget.csv", """poste,prevu,reel
loyer,900,900
courses,400,465
transport,120,98
loisirs,150,210
""")

let r = conv.ask("Quels postes dépassent le budget prévu, et de combien ?",
                 attachments = @[attach("budget.csv")])
echo r.text
```

`attach(chemin)` détecte le type par le contenu (signature) et l'extension.

## 5.3 Pièces jointes créées en mémoire

```nim
# fichier : memoire.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, maxTokens = 300)

# texte construit par le programme
let rapport = attachText("rapport.md", "# Incident 42\nServeur indisponible de 14 h à 14 h 20.\nCause : disque plein.")
# octets bruts (ex. reçus par le réseau) : le type est détecté
let donnees = attachData("mesures.json", """{"temperature": [18.5, 19.2, 21.0], "unite": "°C"}""")

echo conv.ask("Rédige un message d'excuse aux clients à partir du rapport, " &
              "et donne la température moyenne.", attachments = @[rapport, donnees]).text
```

## 5.4 Voir ce que le modèle reçoit

`toPrompt` montre le texte exact inséré. C'est l'outil de diagnostic n° 1 :

```nim
# fichier : apercu.nim
import nimllm

let img = renderSvg("""<svg viewBox="0 0 60 40"><rect width="60" height="40" fill="navy"/>
  <circle cx="30" cy="20" r="12" fill="yellow"/></svg>""", 120, 80)
img.writeBmp("drapeau.bmp")
let pj = attach("drapeau.bmp")
echo pj.kind, " ", pj.width, "x", pj.height
echo pj.toPrompt()
```

```
akImage 120x80
### Pièce jointe : drapeau.bmp (image/bmp)
Image BMP de 120×80 pixels.
Luminosité moyenne : 21 %
Couleurs dominantes : bleu foncé (80 %), jaune (18 %)
Aperçu (48×16, clair = espace, sombre = @) :
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@%...........@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@%...............@@@@@@@@@@@@@@@@
...
```

## 5.5 Documents PDF

```nim
# fichier : pdf.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chemin = if paramCount() >= 1: paramStr(1) else: "facture.pdf"
let pdf = attach(chemin)
echo "Texte extrait : ", pdf.text.len, " caractères"
let conv = newChat(modele, nCtx = 8192)
echo conv.ask("Quel est le montant total et la date d'échéance ?", attachments = @[pdf]).text
```

L'extraction gère les flux compressés (FlateDecode) et les opérateurs de texte
usuels. Limites : PDF scannés (images), polices à encodage personnalisé
(certains PDF produits par des logiciels de PAO), mise en page en colonnes.
En cas de doute, affichez `pdf.text`.

## 5.6 Images : ce qu'il est possible d'en tirer

Le modèle ne voit pas l'image, mais la description fournie suffit pour des
questions simples (couleurs, luminosité, formes grossières, orientation). Pour
une analyse fine, vous pouvez calculer vous-même des informations et les joindre
en texte :

```nim
# fichier : analyse_image.nim
## Calcule des statistiques sur une image et les fait commenter.
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chemin = if paramCount() >= 1: paramStr(1) else: "photo.png"
let pj = attach(chemin)
if pj.image == nil:
  quit "format non décodé (PNG, BMP ou PPM attendus) : " & chemin

# Proportion de pixels « ciel » (bleus) dans la moitié haute
let img = pj.image
var bleus = 0
for y in 0 ..< img.height div 2:
  for x in 0 ..< img.width:
    let (r, g, b) = img.getPixel(x, y)
    if b.int > r.int + 30 and b.int > g.int: inc bleus
let ratio = bleus * 100 div max(1, img.width * img.height div 2)

let conv = newChat(modele)
let info = attachText("mesures.txt", "Pixels bleus dans la moitié haute : " & $ratio & " %")
echo conv.ask("Cette photo est-elle prise en extérieur par beau temps ? Justifie.",
              attachments = @[pj, info]).text
```

Les PNG (8/16 bits, gris, couleur, palette, transparence, non entrelacés) sont
décodés entièrement en Nim (`decodePng`), ainsi que les BMP 24/32 bits et PPM.

## 5.7 Plusieurs documents, comparaison

```nim
# fichier : comparer.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let v1 = attachText("contrat_v1.txt", "Durée : 12 mois. Préavis : 1 mois. Prix : 30 €/mois.")
let v2 = attachText("contrat_v2.txt", "Durée : 24 mois. Préavis : 3 mois. Prix : 27 €/mois.")
let conv = newChat(modele, sampling = greedySampling())
echo conv.ask("Liste les différences entre les deux versions du contrat.",
              attachments = @[v1, v2], format = ofMarkdown).text
```

## 5.8 Bonnes pratiques

* Posez la question **avant** les pièces jointes (c'est ce que fait `ask`) et
  soyez précis sur ce qu'il faut en extraire.
* Pour un gros document, découpez-le (chapitre 7 : résumé « map-reduce » et RAG).
* Les pièces jointes restent dans l'historique : faites `conv.reset()` entre deux
  documents sans rapport pour libérer le contexte.
* Les pièces jointes ne sont jamais interprétées comme des balises de contrôle :
  un document contenant `<|eot_id|>` ne peut pas détourner la conversation.

**Suite :** [06 — Formats de réponse](06-formats-de-reponse.md)
