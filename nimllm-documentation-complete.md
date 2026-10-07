# nimllm

**Bibliothèque de modèles de langage écrite à 100 % en Nim, sans aucune dépendance externe** :
ni llama.cpp, ni Python, ni BLAS, ni bibliothèque C. Seuls le compilateur Nim et un fichier
de modèle au format GGUF (par exemple *Llama 3.2 1B Instruct*) sont nécessaires.

```nim
import nimllm

let modele = loadModel("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let conv = newChat(modele, system = "Tu es un assistant concis.")
echo conv.ask("Quelle est la capitale du Japon ?").text
```

## Ce que fait la bibliothèque

| Domaine | Fonctionnalités |
|---|---|
| **Modèles** | lecture GGUF (mappage mémoire), architectures `llama` (Llama 2/3/3.1/3.2, Mistral, TinyLlama…), `qwen2`, `qwen3` ; poids F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K |
| **Tokeniseurs** | BPE niveau octet (Llama 3, Qwen, GPT-2) et SentencePiece (Llama 2, Mistral), validés sur les jeux de tests de llama.cpp |
| **Dialogue** | prompt système (contexte), historique multi-tours, gabarits (Llama 3, ChatML, Mistral, Llama 2, Gemma, Phi-3), réutilisation du cache KV, gestion du débordement de contexte, sauvegarde/reprise, annuler/régénérer/prolonger |
| **Génération** | flux token par token, température, top-k, top-p, min-p, pénalités, graine, chaînes d'arrêt, biais de logits |
| **Pièces jointes** | texte, code, CSV, JSON, Markdown, HTML, PDF (extraction du texte), PNG/BMP/PPM (couleurs + aperçu), JPEG/GIF/WebP (dimensions), WAV |
| **Formats de réponse** | texte, Markdown, JSON (validé, nouvelles tentatives), **image** (SVG → BMP rastérisé), **audio** (WAV par synthèse vocale), **fichier** |
| **Outils** | embeddings, recherche sémantique (RAG), perplexité, appel d'outils / agents |
| **Apprentissage** | tenseurs à différentiation automatique, AdamW/SGD, planification cosinus, écrêtage, accumulation, points de sauvegarde |
| **Ajustement** | LoRA sur un modèle quantifié existant (Llama 3.2…), application à la volée ou fusion en un nouveau GGUF |
| **Création** | entraînement d'un tokeniseur BPE, architecture Llama configurable, pré-entraînement, ajustement au dialogue, export GGUF (lisible par llama.cpp, Ollama, LM Studio), quantification |

## Démarrage rapide

```sh
# 1. un modèle GGUF (exemple : Llama 3.2 1B Instruct quantifié Q4_K_M, ~0,8 Go)
mkdir -p modeles   # placez-y le fichier .gguf (voir docs/01-installation.md)

# 2. un exemple
nim c -r examples/ex01_bonjour.nim modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# 3. un assistant interactif complet
nim c -r examples/ex04_chat_terminal.nim modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# 4. créer votre propre modèle de A à Z (aucun téléchargement nécessaire)
nim c -r examples/ex17_creer_modele.nim
```

## Documentation

La documentation, dans **`docs/`**, est progressive : chaque chapitre
s'appuie sur le précédent, du premier « bonjour » à la création d'un modèle.

1. **Installation**
2. **Premiers pas**
3. **Conversation et contexte**
4. **Réglages de la génération**
5. **Pièces jointes**
6. **Formats de réponse : texte, JSON, image, audio, fichier**
7. **Cas pratiques : RAG, résumé, agents, extraction…**
8. **Sous le capot : tokens, logits, cache, embeddings**
9. **Bases de l'apprentissage automatique (autograd)**
10. **Ajuster un modèle existant avec LoRA**
11. **Créer un nouveau modèle**
12. **GGUF et quantification**
13. **Performances et dépannage**
14. **Référence de l'API**

Les 20 programmes de [`examples/`](examples/) compilent et s'exécutent tels quels.

## Organisation

```
nimllm/
├── src/nimllm.nim            module principal (importe tout)
├── src/nimllm/
│   ├── gguf.nim              lecture / écriture GGUF
│   ├── quant.nim             formats quantifiés, produits scalaires entiers
│   ├── tokenizer.nim         BPE et SentencePiece
│   ├── model.nim             moteur d'inférence (cache KV, RoPE, GQA, LoRA à la volée)
│   ├── sampler.nim           échantillonnage
│   ├── chat.nim              conversation, formats de réponse
│   ├── attachments.nim       pièces jointes ; inflate.nim (zlib) pour PNG/PDF
│   ├── image.nim             images, dessin, rendu SVG
│   ├── audio.nim             WAV, synthèse vocale, mélodies
│   ├── autograd.nim          tenseurs différentiables
│   ├── nn.nim                Transformer entraînable, LoRA, export
│   ├── train.nim             optimiseurs, données, boucle d'entraînement
│   ├── tokentrain.nim        apprentissage de tokeniseurs BPE
│   └── parallel.nim          pool de threads
├── examples/                 20 programmes commentés
├── docs/                     documentation progressive
└── tests/                    tests (gradients, tokeniseurs, GGUF...)
```

## Fiabilité

Le moteur a été vérifié contre l'implémentation de référence **llama.cpp** :

* tokeniseurs : 100 % des cas de test officiels de llama.cpp (Llama 3, Llama 2/SPM, Qwen 2,
  GPT-2, Phi-3, DeepSeek) produisent exactement les mêmes tokens ;
* déquantification : identique bit à bit à la référence `gguf-py` pour tous les formats ;
* inférence : génération gloutonne identique token pour token à llama.cpp (architectures
  llama et qwen2, formats F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K) ;
* export : les modèles créés ou fusionnés par nimllm se chargent et répondent à
  l'identique dans llama.cpp ;
* différentiation automatique : toutes les opérations sont validées par différences finies.

## Limites (honnêtement)

* **Vitesse** : calcul sur processeur uniquement (pas de GPU). Sur 2 cœurs, un modèle de la
  taille de Llama 3.2 1B Q4_K_M génère environ 5 à 7 tokens/s (llama.cpp, très optimisé à la
  main : ~13). Le traitement du prompt est plus lent que dans llama.cpp.
* **Vision et audio en entrée** : Llama 3.2 1B/3B est un modèle textuel. Les images et sons
  joints lui sont *décrits* (dimensions, couleurs, aperçu, durée), il ne les perçoit pas.
* **Image en sortie** : le modèle écrit du SVG que nimllm rastérise ; la qualité dépend de la
  capacité du modèle à dessiner en SVG (modeste pour un modèle de 1 milliard de paramètres).
* **Audio en sortie** : synthèse vocale par formants, robotique mais autonome.
* **Entraînement** : faisable sur processeur pour de petits modèles et pour LoRA ; un
  pré-entraînement de la taille de Llama 3.2 demanderait des milliers de GPU.
* Architectures non prises en charge : Gemma, Phi-3 (poids fusionnés), Mixture-of-Experts,
  modèles multimodaux ; formats IQ* (i-quants).

## Licence

MIT. Les modèles que vous utilisez ont leur propre licence (par exemple la *Llama 3.2
Community License* de Meta).


---

# Documentation de nimllm

Cette documentation se lit dans l'ordre : chaque chapitre introduit quelques notions
nouvelles et s'appuie sur les précédents. Chaque chapitre contient des programmes
**complets** que vous pouvez copier dans un fichier `.nim` et exécuter tels quels.

## Parcours

| Niveau | Chapitre | Vous saurez… |
|---|---|---|
| Débutant | **01 — Installation** | installer Nim, compiler, obtenir un modèle GGUF |
| Débutant | **02 — Premiers pas** | poser une question, afficher la réponse en direct |
| Débutant | **03 — Conversation et contexte** | définir un rôle, dialoguer, sauvegarder, gérer la mémoire |
| Intermédiaire | **04 — Réglages de la génération** | contrôler créativité, longueur, répétitions, reproductibilité |
| Intermédiaire | **05 — Pièces jointes** | joindre texte, CSV, code, PDF, images, sons |
| Intermédiaire | **06 — Formats de réponse** | obtenir du JSON, une image, un fichier audio, un fichier |
| Intermédiaire | **07 — Cas pratiques** | RAG, résumé de longs documents, agents à outils, classification |
| Avancé | **08 — Sous le capot** | tokens, logits, cache KV, gabarits, embeddings, perplexité |
| Avancé | **09 — Bases de l'apprentissage** | tenseurs, gradients, optimiseurs, réseaux de neurones |
| Expert | **10 — Ajuster un modèle avec LoRA** | spécialiser Llama 3.2 sur vos données |
| Expert | **11 — Créer un nouveau modèle** | tokeniseur, architecture, pré-entraînement, dialogue, export |
| Expert | **12 — GGUF et quantification** | inspecter, écrire, quantifier des fichiers de modèle |
| Tous | **13 — Performances et dépannage** | aller plus vite, comprendre les erreurs |
| Tous | **14 — Référence de l'API** | toutes les fonctions publiques |

## Conventions

* Les programmes commencent par un commentaire `# fichier : nom.nim`.
* Le chemin du modèle est lu sur la ligne de commande ou dans la variable
  d'environnement `NIMLLM_MODELE` :

  ```sh
  export NIMLLM_MODELE=$HOME/modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  nim c -r mon_programme.nim
  ```

* Compilez **toujours** avec `-d:release` (ou utilisez le `config.nims` fourni) :
  en mode debug, le calcul est 10 à 30 fois plus lent.

## Vocabulaire minimal

| Terme | Sens |
|---|---|
| **LLM** | *Large Language Model* : réseau de neurones qui prédit la suite d'un texte |
| **token** | morceau de texte (mot, partie de mot, ponctuation) manipulé par le modèle |
| **contexte** | ensemble des tokens que le modèle « voit » (prompt + historique + réponse) |
| **prompt système** | consigne permanente qui définit le rôle et le comportement du modèle |
| **GGUF** | format de fichier des modèles (poids + tokeniseur + métadonnées) |
| **quantification** | compression des poids (ex. 4 bits au lieu de 32) |
| **logits** | scores bruts attribués par le modèle à chaque token possible |
| **échantillonnage** | façon de choisir le token suivant à partir des logits |
| **LoRA** | technique d'ajustement léger : on n'entraîne que de petites matrices ajoutées |


---

# 01 — Installation

## 1.1 Installer Nim

nimllm demande **Nim 2.0 ou plus récent** et un compilateur C (gcc ou clang).

```sh
# Linux / macOS : installeur officiel
curl https://nim-lang.org/choosenim/init.sh -sSf | sh
# ou via le gestionnaire de paquets : apt install nim / brew install nim
nim -v        # doit afficher 2.x
```

Sous Windows, utilisez l'installeur de <https://nim-lang.org/install.html> (il fournit MinGW).

## 1.2 Installer nimllm

nimllm ne dépend d'aucun paquet. Deux possibilités :

```sh
# a) installation comme paquet Nimble (depuis le dossier du projet)
cd nimllm
nimble install

# b) sans installation : indiquer le chemin des sources à la compilation
nim c -d:release --path:chemin/vers/nimllm/src mon_programme.nim
```

Le dossier `examples/` contient un `config.nims` qui règle tout pour vous
(chemin des sources, `-d:release`, instructions SIMD) :

```sh
nim c -r examples/ex01_bonjour.nim
```

Pour vos propres projets, créez un `config.nims` à côté de vos sources :

```nim
# fichier : config.nims
switch("path", "/chemin/vers/nimllm/src")   # inutile si installé avec nimble
switch("define", "release")                  # indispensable pour la vitesse
switch("passC", "-march=native")             # utilise AVX2/AVX-512 si disponibles
```

> **Important.** Sans `-d:release`, Nim compile en mode debug avec toutes les
> vérifications : l'inférence est alors 10 à 30 fois plus lente.

## 1.3 Obtenir un modèle

nimllm lit les fichiers **GGUF**, le format de llama.cpp, très répandu. Pour
démarrer, le modèle conseillé est **Llama 3.2 1B Instruct en Q4_K_M** (~0,8 Go,
fonctionne avec 2 Go de mémoire). Le 3B (~2 Go) répond nettement mieux, au prix
d'une vitesse divisée par environ 2,5.

Sources possibles :

* **Hugging Face** : cherchez « Llama-3.2-1B-Instruct-GGUF » ; prenez le fichier
  `...Q4_K_M.gguf`. Les dépôts officiels de Meta demandent d'accepter la licence
  *Llama 3.2 Community License*.
* **Ollama** : si vous avez déjà fait `ollama pull llama3.2:1b`, le modèle est un
  fichier GGUF dans `~/.ollama/models/blobs/` (le plus gros fichier `sha256-…`).
  Vous pouvez l'utiliser directement avec nimllm.
* **Votre propre modèle** : le chapitre 11 montre comment en créer un, sans
  rien télécharger.

Autres modèles compatibles : Llama 3.1 8B, Mistral 7B, Qwen 2.5 (0.5B à 7B),
Qwen 3, TinyLlama, SmolLM, etc., en GGUF F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q4_0.

Rangez le modèle, par exemple, dans `modeles/` :

```
mon_projet/
├── config.nims
├── modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf
└── bonjour.nim
```

## 1.4 Vérifier l'installation

```nim
# fichier : verifier.nim
## Affiche les caractéristiques d'un modèle GGUF.
import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin, verbose = true)
echo modele.describe
echo "Threads de calcul : ", numThreads()
echo "Gabarit de dialogue : ", detectTemplate(modele.tokenizer)
```

```sh
nim c -r verifier.nim modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Sortie typique pour Llama 3.2 1B (les valeurs exactes dépendent du fichier) :

```
Modèle Llama 3.2 1B Instruct [llama]
  couches=16 dim=2048 ffn=8192 têtes=32 têtesKV=8 dimTête=64
  vocab=128256 contexte=131072 rope_base=500000.0 paramètres=1235.8 M
  poids : gtQ4_K (attention), gtQ6_K (sortie)
Threads de calcul : 8
Gabarit de dialogue : llama3
```

## 1.5 Mémoire nécessaire

| Élément | Llama 3.2 1B Q4_K_M | Llama 3.2 3B Q4_K_M |
|---|---|---|
| Poids (mappés, partagés) | ~0,8 Go | ~2,0 Go |
| Cache KV par contexte de 4096 tokens | ~0,25 Go | ~0,9 Go |
| Ajustement LoRA (lot de 128 tokens) | +1,3 Go | +3 Go |

Les poids sont *mappés* depuis le fichier : ils ne sont chargés en mémoire qu'à
la demande et plusieurs conversations les partagent.

**Suite :** **02 — Premiers pas**


---

# 02 — Premiers pas

Objectif : poser une question à un modèle et exploiter la réponse.

## 2.1 Trois objets à connaître

```
LlmModel  ──►  Chat (conversation)  ──►  Reply (réponse)
 poids          historique, réglages       texte, fichiers, statistiques
 tokeniseur     cache KV (LlmContext)
```

* `loadModel(chemin)` ouvre le fichier GGUF et retourne un **`LlmModel`**.
* `newChat(modele)` crée une **conversation** (`Chat`) qui garde l'historique.
* `conv.ask(question)` retourne une **`Reply`** ; le texte est dans `.text`.

## 2.2 Le programme minimal

```nim
# fichier : bonjour.nim
import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

let modele = loadModel(chemin)                 # 1. charger le modèle
let conv = newChat(modele)                     # 2. ouvrir une conversation
let r = conv.ask("Quelle est la capitale de la France ?")   # 3. poser la question
echo r.text                                    # 4. utiliser la réponse
```

```sh
nim c -r -d:release bonjour.nim modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf
# La capitale de la France est Paris.
```

## 2.3 Afficher la réponse au fur et à mesure

Pour les réponses longues, on affiche chaque morceau dès qu'il est produit grâce au
paramètre `onToken`. La fonction reçoit un morceau de texte (UTF-8 complet) et
retourne `true` pour continuer.

```nim
# fichier : flux.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, maxTokens = 400)

let r = conv.ask("Explique la photosynthèse à un enfant de 8 ans.",
  onToken = proc (morceau: string): bool =
    stdout.write morceau
    stdout.flushFile()
    true)
echo ""
echo "(", r.completionTokens, " tokens en ", r.seconds.int, " s, ",
     r.tokensPerSecond.int, " tokens/s, arrêt : ", r.stopReason, ")"
```

## 2.4 Ce que contient une réponse

| Champ | Contenu |
|---|---|
| `text` | texte final, nettoyé selon le format demandé |
| `raw` | texte brut tel que produit par le modèle |
| `files` | fichiers créés (images, audio, fichiers) |
| `json` | objet JSON analysé (format JSON) |
| `promptTokens` | taille du prompt envoyé (système + historique + question) |
| `completionTokens` | nombre de tokens générés |
| `stopReason` | `srEndOfText` (fin naturelle), `srMaxTokens`, `srStopString`, `srContextFull`, `srCallback` |
| `seconds`, `tokensPerSecond` | durée et vitesse |

Si `stopReason == srMaxTokens`, la réponse a été coupée : augmentez `maxTokens`
ou appelez `conv.continueReply()` pour la prolonger.

```nim
# fichier : prolonger.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, maxTokens = 40)       # volontairement court
var r = conv.ask("Décris les quatre saisons.")
while r.stopReason == srMaxTokens:
  echo "... (coupé, on prolonge)"
  r = conv.continueReply(maxTokens = 60)
echo r.text                                       # texte complet
```

## 2.5 Gérer les erreurs

Les fonctions lèvent des exceptions explicites :

| Exception | Cause typique |
|---|---|
| `GgufError` | fichier absent, tronqué ou qui n'est pas un GGUF |
| `ModelError` | architecture ou format de poids non pris en charge |
| `ChatError` | message plus long que le contexte, JSON/SVG introuvable dans la réponse |
| `IOError` | pièce jointe introuvable |

```nim
# fichier : erreurs.nim
import nimllm

try:
  let m = loadModel("inexistant.gguf")
  discard m
except GgufError as e:
  echo "Impossible de charger le modèle : ", e.msg
```

## 2.6 Exercices

1. Modifiez `bonjour.nim` pour lire la question sur la ligne de commande
   (`paramStr(2)`).
2. Dans `flux.nim`, interrompez la génération dès que la réponse contient le
   mot « chlorophylle » (retournez `false`).
3. Affichez `r.promptTokens` : pourquoi est-il bien plus grand que le nombre de
   mots de la question ? (indice : chapitre 8, gabarits de dialogue).

**Suite :** **03 — Conversation et contexte**


---

# 03 — Conversation et contexte

Objectif : définir le **contexte** (rôle, consignes), dialoguer sur plusieurs tours,
maîtriser la mémoire de la conversation.

## 3.1 Le prompt système

Le prompt système est une consigne permanente placée avant l'historique. C'est
le moyen principal de définir le comportement du modèle : rôle, ton, langue,
format, règles, connaissances de référence.

```nim
# fichier : role.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

let guide = newChat(modele, system = """
Tu es Léa, guide touristique à Lyon.
- Tu réponds en français, en 3 phrases maximum.
- Tu proposes toujours une adresse précise.
- Si on te parle d'une autre ville, tu ramènes poliment la conversation sur Lyon.""")

echo guide.ask("Où manger une spécialité locale ?").text
echo guide.ask("Et à Marseille ?").text
```

Conseils pour un bon prompt système :

* soyez **explicite** (« 3 phrases maximum » plutôt que « sois bref ») ;
* donnez des **exemples** du format attendu ;
* placez les informations de référence (tarifs, horaires…) dans le prompt système
  ou en pièce jointe ;
* pour un petit modèle (1B), préférez des consignes courtes et simples.

Le prompt système peut être changé à tout moment : `conv.system = "..."`.

## 3.2 Dialogue sur plusieurs tours

Chaque appel à `ask` ajoute la question et la réponse à `conv.history`. Le modèle
voit donc toute la conversation :

```nim
# fichier : dialogue.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, system = "Tu es un professeur de mathématiques patient.")

for question in ["Qu'est-ce qu'un nombre premier ?",
                 "Donne-moi les cinq premiers.",
                 "Et le suivant après le dernier que tu as cité ?"]:
  echo "Élève : ", question
  echo "Prof  : ", conv.ask(question).text, "\n"

# Inspection de l'historique
for m in conv.history:
  echo "[", m.role, "] ", m.content[0 ..< min(60, m.content.len)]
echo conv.stats
```

### Efficacité : le cache KV

Le modèle mémorise les calculs déjà faits sur le début de la conversation (le
*cache clé/valeur*). À chaque tour, nimllm compare le nouveau prompt au contenu
du cache et ne calcule que la partie nouvelle : les longues conversations restent
réactives.

## 3.3 Gérer l'historique

| Opération | Effet |
|---|---|
| `conv.reset()` | efface l'historique (garde le prompt système) |
| `conv.undo()` | retire le dernier échange question/réponse |
| `conv.regenerate()` | remplace la dernière réponse par un nouveau tirage |
| `conv.continueReply()` | prolonge la dernière réponse |
| `conv.add(role, texte)` | ajoute un message sans rien générer |
| `conv.history` | la liste des messages (modifiable directement) |

```nim
# fichier : historique.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele)

discard conv.ask("Propose un prénom pour un chat.")
echo "1er tirage : ", conv.history[^1].content
echo "2e tirage  : ", conv.regenerate().text     # autre proposition
conv.undo()                                       # on oublie cet échange
echo "Messages : ", conv.history.len              # 0

# Injecter un faux historique : utile pour reprendre une session ou
# imposer un style par l'exemple (« few-shot »).
conv.add(roleUser, "Traduis : chat")
conv.add(roleAssistant, "cat")
conv.add(roleUser, "Traduis : chien")
conv.add(roleAssistant, "dog")
echo conv.ask("Traduis : oiseau").text            # bird
```

## 3.4 Sauvegarder et reprendre une conversation

```nim
# fichier : sauvegarde.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

if fileExists("session.json"):
  let conv = newChat(modele)
  conv.loadHistory("session.json")       # système + messages
  echo "Reprise (", conv.history.len, " messages)"
  echo conv.ask("De quoi parlions-nous ?").text
  conv.saveHistory("session.json")
else:
  let conv = newChat(modele, system = "Tu es un coach sportif.")
  echo conv.ask("Je veux courir un 10 km dans 2 mois. Par quoi commencer ?").text
  conv.saveHistory("session.json")
  echo "Session enregistrée ; relancez le programme."
```

Le fichier JSON est lisible et modifiable :

```json
{
  "system": "Tu es un coach sportif.",
  "template": "llama3",
  "messages": [
    {"role": "user", "content": "Je veux courir un 10 km..."},
    {"role": "assistant", "content": "Commence par..."}
  ]
}
```

## 3.5 La fenêtre de contexte

Un modèle ne voit qu'un nombre limité de tokens : la **fenêtre de contexte**.
`newChat(modele, nCtx = 4096)` réserve un cache pour 4096 tokens (Llama 3.2
accepte jusqu’à 131 072, mais chaque token coûte de la mémoire : ~64 Kio pour
Llama 3.2 1B, ~224 Kio pour le 3B).

Que se passe-t-il quand la conversation devient trop longue ?

1. Avant chaque réponse, nimllm vérifie que *prompt + maxTokens* tient dans `nCtx` ;
2. sinon, il retire les **plus anciens** échanges (jamais le prompt système) ;
3. si la dernière question seule est trop longue, `ChatError` est levée.

```nim
# fichier : longue_memoire.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, nCtx = 512, maxTokens = 100)   # petite fenêtre exprès
conv.system = "Retiens tout ce que je te dis."
for i in 1 .. 15:
  discard conv.ask("Note numéro " & $i & " : j'aime le chiffre " & $(i * 7) & ".")
  echo "tour ", i, " : ", conv.history.len, " messages gardés, ", conv.stats
```

Pour garder une mémoire à long terme malgré tout, deux techniques :

* **résumer** périodiquement l'historique et le placer dans le prompt système ;
* stocker les informations et les **retrouver** au besoin (RAG, chapitre 7).

```nim
# fichier : memoire_resumee.nim
## Quand l'historique dépasse 10 messages, on le remplace par un résumé.
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let systemeDeBase = "Tu es un assistant personnel."
let conv = newChat(modele, system = systemeDeBase)

proc compacter(c: Chat) =
  if c.history.len < 10: return
  var transcript = ""
  for m in c.history: transcript.add $m.role & " : " & m.content & "\n"
  let resumeur = newChat(c.model, sampling = greedySampling(), maxTokens = 200)
  let resume = resumeur.ask("Résume les faits importants de cette conversation " &
                            "en une liste courte :\n" & transcript).text
  c.system = systemeDeBase & "\n\nCe que tu sais déjà de l'utilisateur :\n" & resume
  c.reset()

for msg in ["Je m'appelle Paul.", "J'habite à Rennes.", "J'ai deux chats.",
            "Je travaille comme infirmier.", "Mon plat préféré est la galette.",
            "Comment je m'appelle et où j'habite ?"]:
  echo "> ", msg
  echo conv.ask(msg).text
  compacter(conv)
```

## 3.6 Plusieurs conversations en parallèle

Les poids sont partagés : chaque `Chat` n'ajoute que son propre cache. Vous pouvez
donc tenir plusieurs conversations indépendantes avec un seul modèle chargé.

```nim
# fichier : deux_personnages.nim
## Deux personnages dialoguent entre eux.
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let pirate = newChat(modele, system = "Tu es un pirate bourru. Une phrase par réplique.", maxTokens = 60)
let robot = newChat(modele, system = "Tu es un robot très poli. Une phrase par réplique.", maxTokens = 60)

var replique = "Bonjour, qui êtes-vous ?"
for tour in 1 .. 3:
  replique = pirate.ask(replique).text
  echo "Pirate : ", replique
  replique = robot.ask(replique).text
  echo "Robot  : ", replique
```

## 3.7 Exercices

1. Écrivez un correcteur d'orthographe : prompt système « Corrige le texte sans
   commentaire » et `greedySampling()`.
2. Ajoutez à `sauvegarde.nim` une commande pour effacer la session.
3. Mesurez la vitesse du 2e tour d'une conversation (`r.seconds`) avec et sans
   `conv.reset()` entre les tours : observez l'effet du cache.

**Suite :** **04 — Réglages de la génération**


---

# 04 — Réglages de la génération

Objectif : comprendre comment le modèle choisit ses mots et régler créativité,
précision, longueur et reproductibilité.

## 4.1 Comment un token est choisi

À chaque étape, le modèle attribue un score (*logit*) à chacun des ~128 000
tokens de son vocabulaire. Ces scores sont transformés en probabilités, puis un
token est tiré au sort selon une chaîne de filtres :

```
logits ─► pénalités de répétition ─► biais ─► température ─► top-k ─► top-p ─► min-p ─► tirage
```

| Paramètre | Valeur par défaut | Effet |
|---|---|---|
| `temperature` | 0.7 | 0 = toujours le plus probable ; < 1 plus sûr ; > 1 plus inventif |
| `topK` | 40 | ne garde que les K meilleurs candidats (0 = désactivé) |
| `topP` | 0.95 | garde le plus petit groupe totalisant P de probabilité (1 = désactivé) |
| `minP` | 0.05 | élimine les candidats < minP × probabilité du meilleur |
| `repeatPenalty` | 1.1 | > 1 décourage la reprise de tokens récents |
| `repeatLastN` | 64 | fenêtre de tokens surveillés pour la pénalité |
| `presencePenalty` | 0 | pénalité fixe pour tout token déjà apparu |
| `frequencyPenalty` | 0 | pénalité proportionnelle au nombre d'apparitions |
| `seed` | -1 | graine du tirage ; -1 = aléatoire, ≥ 0 = reproductible |
| `logitBias` | vide | ajoute un biais à des tokens précis (-100 = interdit) |

Deux préréglages : `defaultSampling()` (dialogue) et `greedySampling()`
(déterministe : extraction, classification, code, calcul).

## 4.2 Choisir les réglages

```nim
# fichier : temperatures.nim
## Compare la même question à différentes températures.
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let question = "Donne un titre original pour un roman policier."

for t in [0.0, 0.5, 0.9, 1.4]:
  var p = defaultSampling()
  p.temperature = t
  let conv = newChat(modele, sampling = p, maxTokens = 30)
  echo "T=", t, " : ", conv.ask(question).text
```

Repères :

| Usage | Réglage conseillé |
|---|---|
| Extraction, classification, JSON, calcul | `greedySampling()` ou température 0–0.2 |
| Questions/réponses factuelles | température 0.3–0.5 |
| Conversation | `defaultSampling()` (0.7) |
| Écriture créative, brainstorming | température 0.9–1.1, topP 0.95 |
| Au-delà de 1.3 | texte souvent incohérent |

## 4.3 Reproductibilité

Avec une graine fixe, les mêmes entrées donnent toujours la même sortie (sur la
même machine et le même nombre de threads) :

```nim
# fichier : graine.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
var p = defaultSampling()
p.seed = 1234
for essai in 1 .. 2:
  let conv = newChat(modele, sampling = p, maxTokens = 40)
  echo "Essai ", essai, " : ", conv.ask("Invente un proverbe.").text   # identiques
```

## 4.4 Longueur et chaînes d'arrêt

* `maxTokens` limite la longueur (1 token ≈ 0,75 mot en anglais, un peu moins en
  français) ;
* `stop` arrête la génération dès qu'une chaîne apparaît (elle est retirée du
  résultat).

```nim
# fichier : arret.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele)
conv.options.maxTokens = 200
conv.options.stop = @["4."]          # s'arrête avant le 4e point d'une liste
let r = conv.ask("Liste dix fruits, numérotés.")
echo r.text
echo "Raison : ", r.stopReason       # srStopString

# Réglage ponctuel, pour un seul appel :
echo conv.ask("Un mot pour dire bonjour en italien ?", maxTokens = 5).text
```

## 4.5 Éviter les répétitions

Les petits modèles ont tendance à boucler. Leviers, du plus doux au plus fort :

```nim
# fichier : repetitions.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
var p = defaultSampling()
p.repeatPenalty = 1.2        # 1.05 à 1.3
p.repeatLastN = 128          # surveille plus loin
p.frequencyPenalty = 0.3     # pénalise les mots très fréquents
p.presencePenalty = 0.2      # encourage de nouveaux sujets
let conv = newChat(modele, sampling = p, maxTokens = 300)
echo conv.ask("Écris un petit texte sur la mer.").text
```

Attention : une pénalité trop forte dégrade la grammaire (le modèle évite des
mots nécessaires comme « le », « de »).

## 4.6 Interdire ou favoriser des mots

`logitBias` agit sur des **tokens** : on encode d'abord le mot (avec et sans
espace initial, car « Paris » et « ␣Paris » sont des tokens différents).

```nim
# fichier : biais.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = modele.tokenizer

var p = greedySampling()
for variante in ["oui", " oui", "Oui", " Oui"]:
  let ids = tok.encode(variante)
  if ids.len == 1: p.logitBias[ids[0]] = -100.0     # interdit
let conv = newChat(modele, sampling = p, maxTokens = 20)
echo conv.ask("Le ciel est-il bleu ? Réponds par un mot.").text

# Favoriser : un biais positif (+2 à +5) rend un token plus probable.
```

## 4.7 Changer les réglages en cours de conversation

```nim
# fichier : changer.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele)
echo conv.ask("Imagine une créature fantastique.").text   # créatif
conv.setSampling(greedySampling())                         # précis désormais
echo conv.ask("Résume ta description en 5 mots.").text
```

## 4.8 Exercices

1. Générez 5 slogans avec la même question et des graines 1 à 5.
2. Trouvez la température à partir de laquelle Llama 3.2 1B fait des fautes de
   grammaire sur « Raconte ta journée ».
3. Avec `logitBias`, empêchez le modèle d'utiliser la lettre « e » au début des
   mots (exercice difficile : il faut parcourir tout le vocabulaire avec
   `tok.tokenToPiece(id)`).

**Suite :** **05 — Pièces jointes**


---

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

**Suite :** **06 — Formats de réponse**


---

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

**Suite :** **07 — Cas pratiques**


---

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
[`examples/ex13_resume_long.nim`](nimllm/examples/ex13_resume_long.nim). Le point
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
[`examples/ex14_outils.nim`](nimllm/examples/ex14_outils.nim). Schéma :

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

**Suite :** **08 — Sous le capot**


---

# 08 — Sous le capot : tokens, logits, cache, embeddings

Objectif : comprendre et piloter directement le moteur (sans la couche `Chat`).
Ces notions sont nécessaires pour l'entraînement (chapitres 9 à 11).

## 8.1 Le fonctionnement d'un LLM en une page

```
"Le chat dort"
     │  tokeniseur
     ▼
[ 2356, 6369, 294, 10552 ]                 identifiants de tokens
     │  table d'embeddings (vocab × dim)
     ▼
4 vecteurs de 2048 nombres
     │  16 blocs Transformer :
     │    normalisation RMS → attention (qui regarde quels tokens précédents ?)
     │    normalisation RMS → réseau feed-forward (SwiGLU)
     ▼
4 vecteurs « contextualisés »
     │  normalisation + tête de sortie (dim × vocab)
     ▼
logits : 128 256 scores pour le token suivant
     │  échantillonneur
     ▼
token suivant ─► ajouté à l'entrée ─► on recommence
```

Le **cache KV** mémorise, pour chaque token déjà vu, ses clés et valeurs
d'attention : on ne recalcule jamais le passé, seul le nouveau token traverse
le réseau.

## 8.2 Les tokens

```nim
# fichier : tokens.nim
import std/[os, strutils]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = modele.tokenizer

for texte in ["Bonjour", " Bonjour", "anticonstitutionnellement", "2024", "🦙", "<|eot_id|>"]:
  let ids = tok.encode(texte)                       # parseSpecial = true par défaut
  var morceaux: seq[string]
  for id in ids: morceaux.add tok.tokenToPiece(id, renderSpecial = true).escape
  echo alignLeft(texte.escape, 30), " -> ", ids, "  ", morceaux.join(" | ")

# Un texte d'utilisateur ne doit pas créer de tokens de contrôle :
echo tok.encode("<|eot_id|>", parseSpecial = false).len, " tokens (texte ordinaire)"

echo "Vocabulaire : ", tok.vocabSize
echo "Début/fin : ", tok.bosId, " / ", tok.eosId, " ; fins de tour : ", tok.eogIds
echo "Type : ", tok.kind, " (", tok.pre, ")"
```

Points à retenir :

* l'espace fait partie du token (« Bonjour » ≠ « ␣Bonjour ») ;
* un mot rare est découpé en plusieurs tokens ; un emoji en octets ;
* les **tokens spéciaux** (`<|begin_of_text|>`, `<|eot_id|>`…) structurent le
  dialogue ; `parseSpecial = false` empêche un texte de les produire ;
* `tok.eogIds` regroupe les tokens qui terminent une réponse.

## 8.3 Les gabarits de dialogue

Un modèle « Instruct » a été entraîné sur des conversations mises en forme avec
des balises. Pour Llama 3 :

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Salut<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Le modèle complète ensuite le texte et termine par `<|eot_id|>`. nimllm détecte
le gabarit (`detectTemplate`) d'après les métadonnées du GGUF :

| Gabarit | Modèles |
|---|---|
| `tplLlama3` | Llama 3, 3.1, 3.2, 3.3 |
| `tplChatML` | Qwen 2/2.5/3, SmolLM, modèles créés avec nimllm |
| `tplMistral` | Mistral, Mixtral (`[INST] … [/INST]`) |
| `tplLlama2` | Llama 2 Chat (`<<SYS>>`) |
| `tplGemma` | Gemma (gabarit seulement : architecture non prise en charge) |
| `tplPhi3` | Phi-3 (gabarit seulement) |
| `tplRaw` | modèles de base : « Utilisateur : … Assistant : » |

```nim
# fichier : gabarits.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let msgs = @[Message(role: roleSystem, content: "Sois bref."),
             Message(role: roleUser, content: "Salut !")]
for g in [tplLlama3, tplChatML, tplMistral, tplRaw]:
  echo "=== ", g, "\n", renderPrompt(g, msgs)

# Un mauvais gabarit dégrade fortement les réponses : forcez-le si la
# détection échoue (modèle sans métadonnées de chat).
let conv = newChat(modele, templ = tplLlama3)
echo conv.promptTokens().len, " tokens dans le prompt actuel"
```

## 8.4 Évaluer et lire les logits

`LlmContext` est la couche inférieure : un cache KV et des tampons.

```nim
# fichier : logits.nim
import std/[os, strutils, math]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = modele.tokenizer
let ctx = newContext(modele, nCtx = 512)

proc suivants(texte: string) =
  ctx.reset()
  let logits = ctx.eval(tok.encode(texte, addBos = true))
  echo "« ", texte, " » →"
  for (id, p) in topTokens(logits, 5):
    echo "    ", (p * 100).formatFloat(ffDecimal, 1).align(5), " %  ", tok.tokenToPiece(id).escape

suivants("La tour Eiffel se trouve à")
suivants("2 + 2 =")
suivants("Il était une")

# Probabilité d'une suite donnée : somme des log-probabilités de chaque token
proc logProba(debut, suite: string): float =
  ctx.reset()
  var logits = ctx.eval(tok.encode(debut, addBos = true))
  for id in tok.encode(suite):
    var p = logits
    softmaxInPlace(p)
    result += ln(p[id].float)
    logits = ctx.eval([id])

echo "log P(« Paris ») = ", logProba("La capitale de la France est", " Paris").formatFloat(ffDecimal, 2)
echo "log P(« Lyon »)  = ", logProba("La capitale de la France est", " Lyon").formatFloat(ffDecimal, 2)
```

Fonctions utiles du contexte :

| Fonction | Rôle |
|---|---|
| `newContext(modele, nCtx, nBatch)` | crée un contexte (cache de `nCtx` tokens) |
| `ctx.eval(tokens)` | ajoute des tokens, retourne les logits du dernier |
| `ctx.eval(tokens, allLogits = true)` | logits de chaque position (concaténés) |
| `ctx.evalPrompt(tokens)` | comme `eval`, en réutilisant le préfixe déjà en cache |
| `ctx.tokens` / `ctx.nPast` | contenu / taille du cache |
| `ctx.truncate(n)` / `ctx.reset()` | oublie la fin / tout |
| `ctx.lastHidden` | états cachés finaux du dernier lot |

## 8.5 Écrire sa propre boucle de génération

```nim
# fichier : boucle.nim
## Génération manuelle avec un filtre : on interdit les chiffres.
import std/[os, strutils, sequtils]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = modele.tokenizer
let ctx = newContext(modele, nCtx = 1024)
let s = newSampler(defaultSampling())

# Préparation du prompt avec le gabarit
let prompt = encodeMessages(tok, detectTemplate(tok),
  @[Message(role: roleUser, content: "Quel âge a la tour Eiffel ?")])
var logits = ctx.eval(prompt)

var interdits: seq[int]
for id in 0 ..< tok.vocabSize:
  if tok.tokenToPiece(id).anyIt(it.isDigit): interdits.add id

for i in 0 ..< 80:
  for id in interdits: logits[id] = -Inf     # le modèle doit écrire en lettres
  let id = s.sample(logits)
  if id in tok.eogIds: break
  s.accept(id)                               # pour les pénalités de répétition
  stdout.write tok.tokenToPiece(id)
  stdout.flushFile()
  logits = ctx.eval([id])
echo ""
```

## 8.6 Complétion brute et `generate`

`complete` prolonge un texte sans gabarit : utile pour les modèles de base (non
« Instruct ») ou pour imposer un début de réponse.

```nim
# fichier : completion.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let ctx = newContext(modele, nCtx = 1024)
var o = defaultOptions()
o.maxTokens = 60
o.stop = @["\n\n"]
echo ctx.complete("Recette des crêpes\nIngrédients :\n-", o).text

# Imposer le début de la réponse d'un assistant :
let tok = modele.tokenizer
var p = encodeMessages(tok, detectTemplate(tok),
  @[Message(role: roleUser, content: "Cite trois planètes.")])
p.add tok.encode("Voici trois planètes, par ordre alphabétique :", parseSpecial = false)
ctx.reset()
echo ctx.generate(p, o).text
```

## 8.7 Embeddings et similarité

`modele.embed(texte)` retourne un vecteur normalisé (moyenne des états cachés
finaux) : deux textes proches ont un cosinus élevé.

```nim
# fichier : similarite.nim
import std/[os, strutils]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let phrases = ["Le chat dort sur le canapé.", "Un félin fait la sieste sur le sofa.",
               "La bourse de Paris a chuté.", "Les marchés financiers sont en baisse."]
var vecteurs: seq[seq[float32]]
for p in phrases: vecteurs.add modele.embed(p)
for i in 0 ..< phrases.len:
  for j in i+1 ..< phrases.len:
    echo cosineSimilarity(vecteurs[i], vecteurs[j]).formatFloat(ffDecimal, 3), "  ",
         phrases[i], " ↔ ", phrases[j]
```

Les embeddings d'un modèle génératif sont moins discriminants que ceux d'un
modèle spécialisé ; combinez-les avec un score lexical (voir 7.3).

## 8.8 Perplexité

La perplexité mesure à quel point un texte « surprend » le modèle (plus bas =
plus naturel pour lui). Elle sert à comparer des modèles, à mesurer la perte de
qualité due à la quantification, ou à détecter du texte anormal.

```nim
# fichier : perplexite.nim
import std/[os, strutils]
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
for t in ["Le petit chat boit du lait.", "Lait du boit chat petit le.",
          "Colorless green ideas sleep furiously."]:
  echo modele.perplexity(t).formatFloat(ffDecimal, 1).align(9), "  ", t
```

## 8.9 Architecture du moteur

| Élément | Implémentation |
|---|---|
| Poids | mappés depuis le fichier (`memfiles`), jamais copiés |
| Produit matrice-vecteur | activations quantifiées en int8 par blocs + produits entiers (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0), comme ggml ; autres formats : déquantification de la ligne puis produit flottant |
| Parallélisme | pool de threads maison (`parallel.nim`) : lignes de matrices, têtes d'attention |
| Attention | multi-têtes groupée (GQA), causale, cache KV float32 |
| Position | RoPE (paires adjacentes pour llama, moitiés pour qwen) + facteurs Llama 3 (`rope_freqs`) |
| Prompt | traité par lots de `nBatch` tokens (32 par défaut) |
| LoRA | appliqué à la volée (`applyLora`) : y = W·x + échelle·B·(A·x) |

`exactMatmul = true` désactive la quantification des activations (calcul
float32 exact, plus lent), utile pour comparer des résultats.

**Suite :** **09 — Bases de l'apprentissage**


---

# 09 — Bases de l'apprentissage automatique

Objectif : comprendre comment un réseau de neurones apprend, avec le module
`autograd` de nimllm. Ces bases servent ensuite à ajuster (chapitre 10) et créer
(chapitre 11) des modèles de langage. Aucun modèle GGUF n'est nécessaire ici.

## 9.1 Le principe

Apprendre = ajuster des **paramètres** pour faire baisser une **perte** (l'écart
entre ce que le modèle prédit et ce qu'il devrait prédire) :

```
répéter :
  1. propagation avant : calculer la perte à partir des données
  2. rétropropagation  : calculer le gradient de la perte pour chaque paramètre
  3. mise à jour       : déplacer chaque paramètre un peu dans le sens qui réduit la perte
```

Le **gradient** indique dans quelle direction (et avec quelle force) modifier
chaque paramètre. La **différentiation automatique** le calcule pour vous : vous
écrivez seulement le calcul de la perte.

## 9.2 Tenseurs

Un `Tensor` contient des `float32` (`data`), une forme (`shape`) et, s'il est
entraînable, son gradient (`grad`). Les matrices sont rangées ligne par ligne :
un tenseur `[N, C]` contient N lignes de C valeurs.

```nim
# fichier : tenseurs.nim
import std/random
import nimllm

var rng = initRand(1)
let a = fromSeq(@[1'f32, 2, 3, 4, 5, 6], [2, 3])     # matrice 2×3
let z = newTensor([2, 3])                              # zéros
let u = full([3], 1.0)                                 # vecteur de 1
let w = randn([4, 3], std = 0.1, rng)                  # aléatoire N(0, 0.1²), entraînable
echo a                                                 # Tensor@[2, 3] [1.0, 2.0, ... 6.0]
echo a.rows, " lignes × ", a.cols, " colonnes, ", a.numel, " valeurs"
echo w.requiresGrad, " ", z.requiresGrad               # true false

let somme = a + a                 # élément par élément
let produit = a * a
let diffuse = a + u               # un vecteur [C] est ajouté à chaque ligne
let lin = linear(a, w)            # a·wᵀ : [2,3]·[3,4] -> [2,4]
echo somme, "\n", produit, "\n", diffuse, "\n", lin.shape
```

Opérations disponibles :

| Catégorie | Fonctions |
|---|---|
| Arithmétique | `+`, `-`, `*` (élément par élément ou diffusion d'un vecteur ligne), `scale`, `sum`, `mean` |
| Algèbre | `linear(x, w, biais)` (x·wᵀ+b), `matmul(a, b)`, `concatCols` |
| Activations | `relu`, `silu`, `gelu`, `sigmoid`, `tanhT`, `softmax` |
| Normalisation | `rmsnorm(x, gain)` |
| Langage | `embedding(table, ids)`, `rope`, `causalAttention`, `crossEntropy` |
| Pertes | `crossEntropy(logits, cibles)`, `mseLoss(prediction, cible)` |
| Divers | `reshape`, `detach`, `dropout`, `noGrad:` |

## 9.3 Le gradient automatiquement

```nim
# fichier : gradient.nim
import nimllm

# f(x, y) = somme(x * y + x)   =>  df/dx = y + 1 ; df/dy = x
let x = fromSeq(@[1'f32, 2, 3], [3], requiresGrad = true)
let y = fromSeq(@[4'f32, 5, 6], [3], requiresGrad = true)
let f = sum(x * y + x)
backward(f)                       # remplit x.grad et y.grad
echo "f = ", f.item               # 1*4+1 + 2*5+2 + 3*6+3 = 38
echo "df/dx = ", x.grad           # @[5.0, 6.0, 7.0]
echo "df/dy = ", y.grad           # @[1.0, 2.0, 3.0]

# Les gradients s'accumulent : remettez-les à zéro avant chaque étape.
x.zeroGrad(); y.zeroGrad()

# En inférence, désactivez la construction du graphe (plus rapide, moins de mémoire) :
noGrad:
  let g = sum(x * y)
  echo g.item, " (pas de graphe : ", g.requiresGrad, ")"
```

## 9.4 Premier apprentissage : une régression linéaire

On cherche `w` et `b` tels que `y ≈ w·x + b`, à partir d'exemples bruités.

```nim
# fichier : regression.nim
import std/[random, strutils]
import nimllm

var rng = initRand(42)
# Données : y = 2,5·x − 1 + bruit
var xs, ys: seq[float32]
for i in 0 ..< 100:
  let x = rng.rand(4.0) - 2.0
  xs.add float32(x)
  ys.add float32(2.5 * x - 1.0 + rng.gauss() * 0.1)
let X = fromSeq(xs, [100, 1])          # 100 exemples, 1 caractéristique
let Y = fromSeq(ys, [100, 1])

let w = param([1, 1], 0.1, rng, "w")   # paramètres entraînables
let b = full([1], 0, true, "b")
let opt = newSGD(@[w, b], lr = 0.05, momentum = 0.9)

for etape in 1 .. 200:
  opt.zeroGrad()                        # 0. gradients à zéro
  let prediction = linear(X, w, b)      # 1. propagation avant
  let perte = mseLoss(prediction, Y)
  backward(perte)                       # 2. rétropropagation
  opt.update()                          # 3. mise à jour
  if etape mod 40 == 0:
    echo "étape ", etape, "  perte ", perte.item.formatFloat(ffDecimal, 5),
         "  w = ", w.data[0].formatFloat(ffDecimal, 3), "  b = ", b.data[0].formatFloat(ffDecimal, 3)
```

## 9.5 Un vrai réseau de neurones : classer des points

Deux nuages de points enroulés en spirale ne sont pas séparables par une droite :
il faut des couches cachées non linéaires.

```nim
# fichier : spirales.nim
import std/[random, math, strutils]
import nimllm

var rng = initRand(7)
# 1. Données : deux spirales de 100 points
var pts: seq[float32]
var classes: seq[int]
for c in 0 .. 1:
  for i in 0 ..< 100:
    let r = i.float / 100.0
    let t = c.float * PI + r * 4.0 + rng.gauss() * 0.15
    pts.add float32(r * cos(t)); pts.add float32(r * sin(t))
    classes.add c
let X = fromSeq(pts, [200, 2])

# 2. Modèle : 2 -> 32 -> 32 -> 2
let w1 = param([32, 2], 1.0, rng);  let b1 = full([32], 0, true)
let w2 = param([32, 32], 0.2, rng); let b2 = full([32], 0, true)
let w3 = param([2, 32], 0.2, rng);  let b3 = full([2], 0, true)
let params = @[w1, b1, w2, b2, w3, b3]

proc modele(x: Tensor): Tensor =
  let h1 = relu(linear(x, w1, b1))
  let h2 = relu(linear(h1, w2, b2))
  linear(h2, w3, b3)                      # logits des 2 classes

# 3. Entraînement avec AdamW (l'optimiseur des LLM)
let opt = newAdamW(params, lr = 0.01, weightDecay = 0.0)
for etape in 1 .. 1000:
  opt.zeroGrad()
  let perte = crossEntropy(modele(X), classes)
  backward(perte)
  discard clipGradNorm(params, 1.0)       # évite les « explosions » de gradient
  opt.update()
  if etape mod 200 == 0:
    var justes = 0
    noGrad:
      let p = modele(X)
      for i in 0 ..< 200:
        let pred = if p.data[2*i+1] > p.data[2*i]: 1 else: 0
        if pred == classes[i]: inc justes
    echo "étape ", etape, "  perte ", perte.item.formatFloat(ffDecimal, 4),
         "  précision ", justes div 2, " %"
```

## 9.6 Ce qui se passe dans un LLM

Un modèle de langage n'est qu'un réseau plus grand, avec les mêmes briques :

```nim
# fichier : mini_transformer.nim
## Un bloc Transformer écrit à la main avec les opérations d'autograd.
import std/[random, math]
import nimllm

var rng = initRand(3)
const V = 50      # taille du vocabulaire
const D = 32      # dimension des vecteurs
const H = 4       # têtes d'attention
const T = 8       # longueur de séquence

let emb = param([V, D], 0.02, rng)
let normA = full([D], 1, true)
let wq = param([D, D], 0.02, rng); let wk = param([D, D], 0.02, rng)
let wv = param([D, D], 0.02, rng); let wo = param([D, D], 0.02, rng)
let normF = full([D], 1, true)
let w1 = param([4*D, D], 0.02, rng); let w2 = param([D, 4*D], 0.02, rng)
let normS = full([D], 1, true)

var invFreq: seq[float32]
for i in 0 ..< (D div H) div 2: invFreq.add float32(pow(10000.0, -2.0 * i.float / (D div H).float))
var positions: seq[int]
for t in 0 ..< T: positions.add t

proc avant(ids: seq[int]): Tensor =
  var x = embedding(emb, ids)                                  # [T, D]
  let h = rmsnorm(x, normA)
  let q = rope(linear(h, wq), H, D div H, positions, invFreq)
  let k = rope(linear(h, wk), H, D div H, positions, invFreq)
  let v = linear(h, wv)
  x = x + linear(causalAttention(q, k, v, 1, T, H, H, D div H), wo)   # attention + résiduel
  x = x + linear(silu(linear(rmsnorm(x, normF), w1)), w2)              # feed-forward + résiduel
  linear(rmsnorm(x, normS), emb)                                       # logits [T, V] (poids liés)

# Tâche jouet : apprendre à recopier la séquence décalée d'un cran (prédire le token suivant)
let params = @[emb, normA, wq, wk, wv, wo, normF, w1, w2, normS]
let opt = newAdamW(params, lr = 3e-3)
for etape in 1 .. 300:
  var ids: seq[int]
  for t in 0 .. T: ids.add (t * 3 + etape) mod V        # suite arithmétique
  opt.zeroGrad()
  let perte = crossEntropy(avant(ids[0 ..< T]), ids[1 .. T])
  backward(perte)
  opt.update()
  if etape mod 100 == 0: echo "étape ", etape, " perte ", perte.item
```

C'est exactement la structure de `Transformer` (module `nn`), qui ajoute
plusieurs blocs, l'attention groupée, l'export GGUF, LoRA, etc.

## 9.7 Créer sa propre opération

Toute fonction peut devenir différentiable : calculez le résultat, puis
enregistrez la fonction de rétropropagation avec `makeNode`. Vérifiez-la avec
`gradCheck` (différences finies).

```nim
# fichier : operation.nim
import std/[random, math]
import nimllm

proc softplus(a: Tensor): Tensor =
  ## y = ln(1 + eˣ) ; dy/dx = sigmoïde(x)
  result = newTensor(a.shape)
  for i in 0 ..< a.numel: result.data[i] = ln(1 + exp(a.data[i]))
  if needsGrad(a):
    let r = result
    result.makeNode(@[a], proc () =
      a.ensureGrad()
      for i in 0 ..< a.numel:
        a.grad[i] += r.grad[i] * (1 / (1 + exp(-a.data[i]))))

var rng = initRand(5)
let x = randn([4, 5], 1.0, rng)
echo "écart relatif : ", gradCheck(proc (): Tensor = sum(softplus(x) * x), x)   # ~1e-3 ou moins
```

## 9.8 Vocabulaire de l'entraînement

| Terme | Sens | Valeurs typiques |
|---|---|---|
| taux d'apprentissage (`lr`) | taille des pas de mise à jour | 1e-4 à 3e-3 (AdamW) |
| *warmup* | montée progressive du taux au début | 1 à 10 % des étapes |
| décroissance cosinus | baisse progressive du taux | jusqu'à `minLr` ≈ lr/10 |
| *weight decay* | rappel des poids vers 0 (régularisation) | 0 à 0.1 |
| écrêtage (`gradClip`) | limite la norme du gradient | 1.0 |
| lot (`batchSize`) | exemples traités ensemble | 4 à 64 |
| époque | un passage sur toutes les données | — |
| surapprentissage | le modèle récite l'entraînement mais généralise mal | à surveiller avec un jeu de validation |

**Suite :** **10 — Ajuster un modèle avec LoRA**


---

# 10 — Ajuster un modèle existant avec LoRA

Objectif : spécialiser un modèle comme Llama 3.2 sur **vos** données (vocabulaire
métier, style, format de réponse, connaissances d'une entreprise), sur un simple
processeur.

## 10.1 Pourquoi LoRA

Réentraîner tous les poids de Llama 3.2 1B demanderait ~20 Go de mémoire
(poids + gradients + états d'AdamW en float32). **LoRA** (*Low-Rank Adaptation*)
gèle les poids d'origine et ajoute, à côté de certaines matrices W, deux petites
matrices A (r × entrée) et B (sortie × r) :

```
y = W·x  +  (alpha / r) · B·(A·x)
    gelé     appris (r = 8 : ~0,5 % des paramètres)
```

Dans nimllm, les poids gelés restent **quantifiés et mappés** depuis le GGUF : la
mémoire nécessaire est celle de l'inférence plus les activations du lot en cours
(environ 10 Mo par token de lot pour Llama 3.2 1B).

| | Ajustement complet (`tmFull`) | LoRA (`tmLora`) |
|---|---|---|
| Paramètres appris | tous | 0,1 à 2 % |
| Mémoire (Llama 3.2 1B) | ~20 Go | ~2 Go (lot de 128 tokens) à ~6 Go (512 tokens) |
| Fichier produit | modèle complet | adaptateur de quelques Mo |
| Usage conseillé | petits modèles (< 50 M) | modèles existants (1B, 3B, 8B) |

**Ce que LoRA fait bien** : imposer un style, un format, un ton, un vocabulaire,
apprendre des réponses types. **Ce qu'il fait moins bien** : ajouter beaucoup de
connaissances nouvelles (préférez alors le RAG, chapitre 7.3, éventuellement
combiné).

## 10.2 Préparer les données

Un fichier **JSONL** : une conversation par ligne. Trois formats sont acceptés :

```json
{"messages": [{"role": "user", "content": "Êtes-vous ouverts le lundi ?"}, {"role": "assistant", "content": "Non, nous sommes fermés le lundi."}]}
{"prompt": "Où êtes-vous situés ?", "response": "Au 12 rue des Lilas, à Nantes."}
{"instruction": "Donne l'adresse e-mail.", "input": "", "output": "contact@boulangerie-dupont.fr"}
```

Conseils :

* **qualité > quantité** : 50 à 500 exemples soignés suffisent souvent ;
* variez les formulations des questions ;
* mettez exactement le style de réponse voulu (longueur, ton, format) ;
* gardez 5 à 10 % des exemples pour la **validation** ;
* seules les réponses de l'assistant sont apprises (la perte ignore le reste).

Exemple fourni : [`examples/donnees/faq_boulangerie.jsonl`](nimllm/examples/donnees/faq_boulangerie.jsonl).

## 10.3 Le programme complet

```nim
# fichier : lora_boulangerie.nim
## nim c -r lora_boulangerie.nim modele.gguf donnees.jsonl
import std/[os, math, strutils]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let fichier = if paramCount() >= 2: paramStr(2) else: "faq_boulangerie.jsonl"
let systeme = "Tu es l'assistant de la boulangerie Dupont."

# 1. Modèle en mode LoRA
var lc = defaultLora()           # rang 8, alpha 16, couches q,k,v,o
lc.rank = 16
lc.alpha = 32
lc.targets = @["q", "k", "v", "o", "gate", "up", "down"]
let m = loadForTraining(chemin, tmLora, lc)
echo "Paramètres appris : ", m.parameterCount

# 2. Données (même gabarit que le modèle, même prompt système qu'à l'usage)
let tout = loadChatJsonl(fichier, m.tokenizer, detectTemplate(m.tokenizer), system = systeme)
let (train, valid) = tout.split(0.1)
var plusLong = 0
for ex in tout.examples: plusLong = max(plusLong, ex.tokens.len)

# 3. Réglages
var tc = defaultTrainConfig()
tc.steps = 200
tc.batchSize = 1              # 1 séquence à la fois : mémoire minimale
tc.gradAccum = 8              # ... mais gradients accumulés sur 8 exemples
tc.seqLen = min(plusLong, 256)
tc.lr = 1e-3
tc.warmup = 20
tc.evalEvery = 50
tc.logEvery = 10
tc.saveEvery = 100
tc.savePath = "boulangerie"   # -> boulangerie.lora.gguf (+ état de l'optimiseur)

# 4. Entraînement
echo "Perte de validation initiale : ", m.evaluate(valid, 2, tc.batchSize, tc.seqLen).formatFloat(ffDecimal, 3)
discard m.train(train, tc, valData = valid)
m.saveLora("boulangerie.lora.gguf")

# 5. Test immédiat
let base = loadModel(chemin)
base.applyLora("boulangerie.lora.gguf")
let conv = newChat(base, system = systeme, sampling = greedySampling(), maxTokens = 80)
for q in ["Vous êtes ouverts lundi ?", "C'est combien la baguette ?"]:
  conv.reset()
  echo q, "\n→ ", conv.ask(q).text
```

Version commentée dans les exemples : [`examples/ex18_lora.nim`](nimllm/examples/ex18_lora.nim).

## 10.4 Lire les courbes

Le journal affiche à chaque ligne :

```
étape    50 | perte 1.2345 | val 1.3012 | lr 9.51e-04 | ‖g‖ 0.82 | 410 tok/s
```

* **perte** (entraînement) doit baisser ; vers 0 = le modèle récite les exemples ;
* **val** (validation) : si elle **remonte** alors que la perte baisse, c'est du
  surapprentissage → moins d'étapes, rang plus faible, plus de données ;
* **‖g‖** : norme du gradient ; des valeurs très grandes et instables → baissez `lr` ;
* une perte qui stagne dès le début → augmentez `lr` (LoRA supporte 1e-4 à 3e-3).

## 10.5 Choisir les hyper-paramètres

| Paramètre | Effet | Point de départ |
|---|---|---|
| `rank` | capacité de l'adaptateur | 8 (style), 16–32 (contenu) |
| `alpha` | intensité de l'adaptation | 2 × rank |
| `targets` | couches adaptées | `q,k,v,o` (léger) ; + `gate,up,down` (plus efficace) |
| `lr` | vitesse d'apprentissage | 1e-3 |
| `steps` | durée | 2 à 5 passages sur les données |
| `seqLen` | longueur max des exemples | longueur du plus long exemple |

Nombre de passages sur les données ≈ `steps × batchSize × gradAccum / nombre d'exemples`.

## 10.6 Utiliser l'adaptateur

Deux possibilités :

```nim
# fichier : utiliser_lora.nim
import std/os
import nimllm

let chemin = getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

# a) À la volée : le modèle de base est inchangé, on peut changer d'adaptateur.
let m = loadModel(chemin)
m.applyLora("boulangerie.lora.gguf")
echo newChat(m).ask("Vos horaires ?").text
m.removeLora()                                   # retour au modèle d'origine

# b) Fusion : un nouveau GGUF autonome (pas de surcoût à l'inférence).
#    Les tenseurs modifiés sont re-quantifiés en Q8_0 (ou gtQ4_K, gtF16...).
mergeLora(chemin, "boulangerie.lora.gguf", "llama-boulangerie.gguf", outType = gtQ8_0)
let fusion = loadModel("llama-boulangerie.gguf")
echo newChat(fusion).ask("Vos horaires ?").text
```

Le GGUF fusionné est un fichier standard : utilisable avec llama.cpp,
LM Studio ou Ollama (`FROM ./llama-boulangerie.gguf` dans un `Modelfile`).

## 10.7 Reprendre ou poursuivre un ajustement

`saveEvery`/`saveCheckpoint` écrivent l'adaptateur et l'état de l'optimiseur.
Pour poursuivre plus tard :

```nim
# fichier : reprendre_lora.nim
import std/os
import nimllm

let chemin = getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
var lc = defaultLora()
lc.rank = 16; lc.alpha = 32
lc.targets = @["q", "k", "v", "o", "gate", "up", "down"]
let m = loadForTraining(chemin, tmLora, lc)

# recharge les matrices A et B sauvegardées
let g = openGguf("boulangerie.lora.gguf")
for p in m.parameters:
  if p.name in g.tensors: p.data = g.tensorF32(p.name)
g.close()

let opt = newAdamW(m.parameters, lr = 5e-4)
if fileExists("boulangerie.optim.gguf"): opt.loadState("boulangerie.optim.gguf")
let ds = loadChatJsonl("faq_boulangerie.jsonl", m.tokenizer, detectTemplate(m.tokenizer),
                       system = "Tu es l'assistant de la boulangerie Dupont.")
var tc = defaultTrainConfig()
tc.steps = 50; tc.batchSize = 1; tc.gradAccum = 8; tc.seqLen = 256; tc.lr = 5e-4; tc.evalEvery = 0
discard m.train(ds, tc, opt = opt)
m.saveLora("boulangerie.lora.gguf")
```

## 10.8 Durée sur processeur

Un pas d'entraînement LoRA coûte environ 3 fois le traitement des mêmes tokens
en inférence. Mesure de référence pour un modèle de la taille de Llama 3.2 1B
(Q4_K_M), sur un processeur à **2 cœurs** : ~30 s par étape pour un lot de 128
tokens (≈ 4 tokens appris par seconde), pic mémoire ≈ 2,2 Go. Le temps est
proportionnel au nombre de tokens du lot et diminue avec le nombre de cœurs.

Exemple : 200 dialogues de 100 tokens, 3 passages = 60 000 tokens ≈ 4 h sur 2
cœurs, ≈ 1 h sur 8 cœurs. Pour des essais rapides, commencez avec peu d'exemples
et peu d'étapes, et validez toute votre chaîne sur le petit modèle du chapitre 11
(quelques secondes).

## 10.9 Ajustement complet d'un petit modèle

Pour un modèle de quelques millions de paramètres (comme ceux du chapitre 11),
tous les poids peuvent être entraînés :

```nim
# fichier : ajustement_complet.nim
import nimllm

let m = loadForTraining("mini-assistant-f32.gguf", tmFull)   # tous les poids en float32
echo m.parameterCount, " paramètres entraînables"
var convs = @[@[Message(role: roleUser, content: "Quelle est la capitale du Pérou ?"),
                Message(role: roleAssistant, content: "La capitale du Pérou est Lima.")]]
let ds = newChatDataset(m.tokenizer, tplChatML, convs)
var tc = defaultTrainConfig()
tc.steps = 100; tc.batchSize = 4; tc.seqLen = 48; tc.lr = 1e-3; tc.evalEvery = 0
discard m.train(ds, tc)
m.saveGguf("mini-assistant-v2.gguf")
```

**Suite :** **11 — Créer un nouveau modèle**


---

# 11 — Créer un nouveau modèle

Objectif : concevoir et entraîner un modèle de langage **à partir de zéro**, puis
l'utiliser exactement comme Llama 3.2 (et même dans llama.cpp ou Ollama).

Le programme complet de ce chapitre est
[`examples/ex17_creer_modele.nim`](nimllm/examples/ex17_creer_modele.nim) : en deux
minutes environ sur un processeur, il produit un mini-assistant qui répond à des
questions de géographie, d'arithmétique et de calendrier.

## 11.1 Les étapes

```
1. données        textes bruts (pré-entraînement) + dialogues (ajustement)
2. tokeniseur     trainBpe  ─────────────►  vocabulaire
3. architecture   newModelConfig + newTransformer
4. pré-entraînement : prédire le mot suivant sur du texte brut
5. ajustement au dialogue (SFT) : apprendre à répondre
6. évaluation     perte de validation, perplexité, essais
7. export         saveGguf (+ quantizeModel)
8. utilisation    loadModel + newChat, comme n'importe quel modèle
```

Ordre de grandeur des tailles :

| Modèle | Paramètres | Données d'entraînement | Matériel |
|---|---|---|---|
| exemple de ce chapitre | 1 M | quelques Ko | 1 CPU, minutes |
| petit modèle spécialisé | 10–50 M | 10–500 Mo de texte | CPU, heures à jours |
| Llama 3.2 1B | 1,2 G | ~9 000 milliards de tokens | milliers de GPU |

Un modèle créé ici ne sait que ce que contiennent vos données : c'est idéal pour
un domaine étroit (commandes d'un appareil, FAQ fermée, langage formel, jeu…).

## 11.2 Le tokeniseur

```nim
# fichier : etape_tokeniseur.nim
import nimllm

let textes = @["Le chat dort. Le chien court. Le chat mange.",
               "Une pomme est rouge. Une banane est jaune."]
let tok = trainBpe(textes,
  vocabSize = 400,                       # 256 octets + fusions + spéciaux
  specials = @["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"],
  minFreq = 2)                           # fusion seulement si la paire apparaît ≥ 2 fois
echo tok.vocabSize, " tokens ; « Le chat dort. » -> ", tok.encode("Le chat dort.")
echo "Gabarit enregistré : ", tok.chatTemplate.len > 0      # ChatML (llama.cpp le lira)
```

Taille du vocabulaire : 256 octets au minimum ; 1 000–8 000 pour un petit
modèle ; Llama 3 utilise 128 256 tokens. Un vocabulaire plus grand raccourcit les
séquences mais grossit la table d'embeddings.

Les tokens `<|im_start|>`/`<|im_end|>` activent le gabarit **ChatML** :
`newChat` le détecte automatiquement.

## 11.3 L'architecture

```nim
# fichier : etape_architecture.nim
import nimllm

let tok = byteLevelTokenizer()            # tokeniseur octet par octet (aucun apprentissage)
let cfg = newModelConfig(
  vocab = tok.vocabSize,
  dim = 256,          # largeur des vecteurs (embedding_length)
  layers = 6,         # nombre de blocs Transformer
  heads = 8,          # têtes d'attention (dim divisible par heads)
  kvHeads = 2,        # têtes clé/valeur (GQA ; heads multiple de kvHeads)
  hidden = 0,         # taille du feed-forward (0 = ≈ 8/3 × dim)
  ctx = 256,          # contexte maximal
  ropeBase = 10000,
  tied = true,        # tête de sortie partagée avec les embeddings
  name = "mon-modele")
let m = newTransformer(cfg, tok, seed = 1)
echo m.parameterCount, " paramètres"
echo "dimension par tête : ", cfg.headDim, ", feed-forward : ", cfg.hidden
```

Règles pratiques :

* `dim / heads` = 32 à 128 ;
* pour l'export en Q4_K/Q6_K, prenez `dim` et `hidden` multiples de 256
  (sinon ces tenseurs sont stockés en F16, ce qui reste valide) ;
* doubler `dim` multiplie le coût par ~4 ; doubler `layers` par ~2.

## 11.4 Pré-entraînement

On apprend la langue en prédisant le token suivant sur du texte brut.

```nim
# fichier : etape_pretrain.nim
import std/[os, strutils]
import nimllm

# Corpus : un ou plusieurs fichiers texte (séparez les documents par 3 sauts de ligne)
let texte = if paramCount() >= 1: readFile(paramStr(1))
            else: "Il était une fois un petit village au bord de la mer. ".repeat(300)
let tok = trainBpe([texte], vocabSize = 1000)
let m = newTransformer(newModelConfig(tok.vocabSize, dim = 128, layers = 4, heads = 4, ctx = 128), tok)

let (train, valid) = newTextDataset(tok, texte).split(0.05)
echo train.len, " tokens d'entraînement"

var tc = defaultTrainConfig()
tc.steps = 300
tc.batchSize = 8
tc.seqLen = 128
tc.lr = 1e-3            # 1e-3 à 3e-3 pour un petit modèle ; 3e-4 pour un plus grand
tc.minLr = 1e-4
tc.warmup = 50
tc.weightDecay = 0.1
tc.evalEvery = 100
tc.sampleEvery = 100    # affiche un échantillon de texte généré
tc.samplePrompt = "Il était"
discard m.train(train, tc, valData = valid)
m.saveGguf("pretrain.gguf")
```

Repère de lecture de la perte (entropie croisée, en nats) : `ln(vocab)` au départ
(6,9 pour 1 000 tokens) ; sous 2 le texte devient lisible ; sous 1 sur un petit
corpus, le modèle commence à réciter.

## 11.5 Ajustement au dialogue (SFT)

On montre au modèle des conversations : il apprend à répondre (seules les
réponses comptent dans la perte).

```nim
# fichier : etape_sft.nim
import nimllm

let m = loadForTraining("pretrain.gguf", tmFull)      # repart du modèle pré-entraîné
var dialogues: seq[seq[Message]]
for (q, r) in [("Bonjour !", "Bonjour ! Que puis-je faire pour toi ?"),
               ("Où est le village ?", "Le village est au bord de la mer.")]:
  dialogues.add @[Message(role: roleUser, content: q),
                  Message(role: roleAssistant, content: r)]
let ds = newChatDataset(m.tokenizer, tplChatML, dialogues)
# (ou : loadChatJsonl("dialogues.jsonl", m.tokenizer, tplChatML))

var tc = defaultTrainConfig()
tc.steps = 300; tc.batchSize = 8; tc.seqLen = 64; tc.lr = 1e-3; tc.evalEvery = 0
discard m.train(ds, tc)
m.saveGguf("assistant.gguf")
```

Pour que le modèle garde ses connaissances générales, mélangez quelques textes
du pré-entraînement aux dialogues, ou utilisez un taux d'apprentissage plus bas.

## 11.6 Évaluer

```nim
# fichier : etape_evaluation.nim
import std/[math, strutils]
import nimllm

let lm = loadModel("assistant.gguf")
echo "Perplexité : ", lm.perplexity("Le village est au bord de la mer.").formatFloat(ffDecimal, 2)
let conv = newChat(lm, sampling = greedySampling(), maxTokens = 40, nCtx = 128)
for q in ["Bonjour !", "Où est le village ?", "Quelle heure est-il ?"]:
  conv.reset()
  echo q, " -> ", conv.ask(q).text
```

Testez toujours avec des questions **absentes** des données d'entraînement : c'est
la seule façon de mesurer la généralisation.

## 11.7 Exporter, quantifier, partager

```nim
# fichier : etape_export.nim
import nimllm

let m = loadForTraining("assistant.gguf", tmFull)
m.saveGguf("assistant-f16.gguf", gtF16)          # moitié de la taille, sans perte notable
quantizeModel("assistant.gguf", "assistant-q8.gguf", gtQ8_0)
quantizeModel("assistant.gguf", "assistant-q4k.gguf", gtQ4_K)
```

Le fichier contient l'architecture (`llama`), le tokeniseur (BPE, pré-découpage
Llama 3), le gabarit ChatML et les poids : il fonctionne avec nimllm, avec
**llama.cpp** (`llama-cli -m assistant-q8.gguf`), et avec **Ollama** :

```
# Modelfile
FROM ./assistant-q8.gguf
```

```sh
ollama create mon-assistant -f Modelfile
ollama run mon-assistant
```

## 11.8 Conseils pour aller plus loin

* **Données** : c'est le facteur n° 1. Nettoyez, dédoublonnez, équilibrez.
* **Échelle** : à budget de calcul fixé, mieux vaut un modèle un peu plus petit
  entraîné sur plus de données (environ 20 tokens de données par paramètre).
* **Taux d'apprentissage** : si la perte explose ou devient `nan`, divisez `lr`
  par 3 ; si elle baisse très lentement, multipliez-le par 2.
* **Contexte** : `seqLen` ≤ `ctx` ; des séquences plus longues coûtent plus cher
  (l'attention est quadratique).
* **Reprise** : `saveEvery` + `saveCheckpoint` + `Optimizer.loadState`
  ([`examples/ex20_reprise_entrainement.nim`](nimllm/examples/ex20_reprise_entrainement.nim)).
* **Distillation** : générez des dialogues avec un grand modèle (chapitre 7.6)
  pour entraîner un petit modèle spécialisé.

**Suite :** **12 — GGUF et quantification**


---

# 12 — GGUF et quantification

Objectif : comprendre, inspecter et produire les fichiers de modèles.

## 12.1 Structure d'un fichier GGUF

```
┌──────────────────────────────┐
│ "GGUF", version, compteurs   │
├──────────────────────────────┤
│ métadonnées clé/valeur       │  general.architecture = "llama"
│                              │  llama.block_count = 16
│                              │  tokenizer.ggml.tokens = [...]
│                              │  tokenizer.chat_template = "..."
├──────────────────────────────┤
│ descriptions des tenseurs    │  nom, dimensions, type, position
├──────────────────────────────┤
│ données (alignées sur 32 o.) │  poids quantifiés
└──────────────────────────────┘
```

Noms des tenseurs (convention llama.cpp) : `token_embd.weight`,
`blk.N.attn_norm.weight`, `blk.N.attn_q/k/v/output.weight`,
`blk.N.ffn_norm.weight`, `blk.N.ffn_gate/up/down.weight`,
`output_norm.weight`, `output.weight` (absent si poids liés),
`rope_freqs.weight` (Llama 3.1+).

## 12.2 Inspecter

```nim
# fichier : inspecter.nim
import std/[os, strutils, tables]
import nimllm

let g = openGguf(if paramCount() >= 1: paramStr(1)
                 else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
echo g.describe(maxTensors = 12)

let arch = g.getStr("general.architecture")
echo "Couches : ", g.getInt(arch & ".block_count")
echo "Contexte : ", g.getInt(arch & ".context_length")
echo "Vocabulaire : ", g.getStrArray("tokenizer.ggml.tokens").len
echo "Gabarit :\n", g.getStr("tokenizer.chat_template")[0 ..< min(300, g.getStr("tokenizer.chat_template").len)]

var parType = initCountTable[GgmlType]()
for nom, t in g.tensors: parType.inc(t.typ, t.byteSize)
for typ, octets in parType:
  echo align($typ, 8), " : ", octets div (1024*1024), " Mio"
g.close()
```

## 12.3 Les formats de poids

| Type | Bits/poids | Bloc | Qualité | Usage |
|---|---|---|---|---|
| F32 | 32 | 1 | référence | entraînement, petits tenseurs |
| F16 / BF16 | 16 | 1 | quasi identique | export sans perte notable |
| Q8_0 | 8,5 | 32 | excellente | défaut sûr, fusion LoRA |
| Q6_K | 6,56 | 256 | très bonne | têtes de sortie |
| Q5_K | 5,5 | 256 | bonne | compromis |
| Q4_K | 4,5 | 256 | correcte | **le plus courant** (Q4_K_M) |
| Q5_0 / Q5_1 / Q4_0 / Q4_1 | 5,5 / 6 / 4,5 / 5 | 32 | correcte | anciens formats |
| Q3_K / Q2_K | 3,4 / 2,6 | 256 | dégradée | lecture seule dans nimllm |

« Q4_K_M » désigne un *mélange* : la plupart des matrices en Q4_K, certaines
(sortie, `attn_v`, `ffn_down`) en Q6_K.

Principe d'un bloc Q8_0 : 32 valeurs → une échelle `d` (float16) + 32 entiers de
−127 à 127 ; valeur ≈ d × q. Les formats « K » regroupent 256 valeurs en
sous-blocs avec des échelles elles-mêmes quantifiées.

```nim
# fichier : formats.nim
import std/[math, strutils]
import nimllm

var v = newSeq[float32](512)
for i in 0 ..< v.len: v[i] = float32(sin(i.float * 0.1) + 0.3 * cos(i.float * 0.7))
for t in [gtF16, gtBF16, gtQ8_0, gtQ6_K, gtQ5_K, gtQ5_0, gtQ4_K, gtQ4_0]:
  let octets = quantize(t, v)                        # float32 -> blocs
  let retour = dequantRow(t, unsafeAddr octets[0], v.len)   # blocs -> float32
  var err = 0.0
  for i in 0 ..< v.len: err += (v[i] - retour[i]).float ^ 2
  echo align($t, 7), ": ", align($octets.len, 5), " octets, erreur RMS ",
       sqrt(err / v.len.float).formatFloat(ffScientific, 2)
```

## 12.4 Quantifier un modèle

```nim
# fichier : quantifier.nim
## nim c -r quantifier.nim entree.gguf sortie.gguf q4_k
import std/os
import nimllm

let entree = paramStr(1)
let sortie = paramStr(2)
let t = if paramCount() >= 3: parseGgmlType(paramStr(3)) else: gtQ8_0
quantizeModel(entree, sortie, t, keepOutput = true)
# keepOutput : embeddings et tête de sortie restent en Q8_0 (meilleure qualité)
echo getFileSize(entree) div (1024*1024), " Mio -> ", getFileSize(sortie) div (1024*1024), " Mio"
echo "perplexité avant : ", loadModel(entree).perplexity("Un texte de test représentatif.")
echo "perplexité après : ", loadModel(sortie).perplexity("Un texte de test représentatif.")
```

Remarques :

* les vecteurs (normes, biais) restent toujours en F32 ;
* un tenseur dont la longueur de ligne n'est pas multiple du bloc (32 ou 256)
  est stocké en F16 ;
* les quantificateurs de nimllm sont des versions simples (min/max par bloc) :
  pour quantifier un *grand* modèle avec la meilleure qualité possible,
  l'outil `llama-quantize` (avec matrice d'importance) reste préférable ; les
  fichiers qu'il produit se lisent dans nimllm.

## 12.5 Écrire ses propres fichiers GGUF

`GgufWriter` permet d'enregistrer n'importe quels tableaux (vecteurs d'un index
RAG, poids d'un réseau maison…) avec des métadonnées :

```nim
# fichier : ecrire_gguf.nim
import nimllm

let w = newGgufWriter()
w.setKV("general.architecture", gStr("index-rag"))
w.setKV("index.nombre", gU32(3))
w.setKV("index.sources", gArrStr(["a.txt", "b.txt", "c.txt"]))
w.addTensorF32("vecteurs", @[4, 3], @[1'f32, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0])  # 3 lignes de 4
w.write("index.gguf")

let g = openGguf("index.gguf")
echo g.getStrArray("index.sources"), " ", g.tensors["vecteurs"].dims
echo g.tensorF32("vecteurs")
g.close()
```

`dims` suit la convention ggml : `dims[0]` est la longueur d'une ligne.

## 12.6 Compatibilité

| Écrit par nimllm | Lu par |
|---|---|
| modèles (`saveGguf`, `mergeLora`, `quantizeModel`) | nimllm, llama.cpp, Ollama, LM Studio, GPT4All… |
| adaptateurs LoRA (`saveLora`) | nimllm (`applyLora`) — format propre, utilisez `mergeLora` pour les autres outils |

| Lu par nimllm | Condition |
|---|---|
| GGUF v2/v3 | architectures llama, mistral, qwen2, qwen3 |
| types F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K…Q6_K | les types IQ* ne sont pas pris en charge |

**Suite :** **13 — Performances et dépannage**


---

# 13 — Performances et dépannage

## 13.1 Options de compilation

| Option | Effet | Recommandation |
|---|---|---|
| `-d:release` | optimisations du compilateur C | **indispensable** |
| `--passC:-march=native` | instructions SIMD du processeur (AVX2, AVX-512…) | conseillé (+20 à 40 %) ; l'exécutable ne tourne alors que sur des processeurs équivalents |
| `-d:danger` | retire toutes les vérifications d'exécution | gain faible (les noyaux de calcul les désactivent déjà) ; déconseillé |
| `--threads:on` | threads (actif par défaut en Nim 2) | ne pas désactiver |

Exemple de `config.nims` de production :

```nim
switch("define", "release")
switch("passC", "-march=native")
switch("opt", "speed")
```

## 13.2 Threads

```nim
# fichier : threads.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
for n in [1, 2, 4, 8]:
  setThreads(n)                           # change le pool à tout moment
  let conv = newChat(modele, sampling = greedySampling(), maxTokens = 32)
  let r = conv.ask("Compte de un à dix.")
  echo n, " thread(s) : ", r.tokensPerSecond.int, " tok/s"
```

* Par défaut, nimllm utilise tous les cœurs logiques (`setThreads(0)`).
* La génération est limitée par la **bande passante mémoire** : au-delà du nombre
  de cœurs physiques, le gain est faible.
* `loadModel(chemin, threads = n)` fixe aussi le nombre de threads.

## 13.3 Ordres de grandeur

Mesures sur un processeur à **2 cœurs** (machine virtuelle), modèle aux
dimensions de Llama 3.2 1B en Q4_K_M, `-d:release --passC:-march=native` :

| Opération | nimllm | llama.cpp (même machine) |
|---|---|---|
| Génération | ~7 tokens/s | ~13 tokens/s |
| Traitement du prompt | ~12 tokens/s | nettement plus rapide (calcul matriciel par lots optimisé) |
| Entraînement LoRA | ~4 tokens/s | — |
| Entraînement d'un modèle de 1 M de paramètres | ~3 000 tokens/s | — |

La vitesse évolue à peu près proportionnellement au nombre de cœurs physiques et
inversement à la taille du modèle (3B ≈ 2,5 fois plus lent que 1B ; Q8_0 ≈ 1,5 fois
plus lent que Q4_K_M).

## 13.4 Conseils de vitesse

1. **Modèle quantifié** : Q4_K_M est le meilleur compromis ; F16/F32 sont lents.
2. **Contexte réutilisé** : gardez la même conversation plutôt que de reconstruire
   le prompt ; un prompt système fixe n'est calculé qu'une fois.
3. **Prompt court** : chaque token du prompt coûte du calcul ; résumez les
   historiques longs (chapitre 3.5).
4. **`maxTokens` adapté** et **chaînes d'arrêt** pour ne pas générer inutilement.
5. **Pièces jointes ciblées** (RAG) plutôt que des documents entiers.
6. **`nBatch`** (`newContext(m, nBatch = 64)`) : lots de prompt plus grands,
   légèrement plus rapides, un peu plus de mémoire.

## 13.5 Problèmes fréquents

| Symptôme | Cause probable | Solution |
|---|---|---|
| Extrêmement lent | compilé sans `-d:release` | ajoutez `-d:release` |
| `GgufError: fichier introuvable` | chemin erroné | vérifiez le chemin / `NIMLLM_MODELE` |
| `GgufError: ce n'est pas un fichier GGUF` | fichier `.safetensors`, `.bin`, téléchargement HTML | prenez la version GGUF du modèle |
| `type de tenseur inconnu … IQ*` | quantification « i-quant » | prenez Q4_K_M, Q5_K_M, Q8_0… |
| `architecture non prise en charge` | Gemma, Phi-3, MoE… | utilisez Llama, Mistral ou Qwen |
| Réponses incohérentes, mélange de balises | mauvais gabarit | forcez `newChat(m, templ = tplLlama3)` (ou le bon) ; vérifiez avec `renderPrompt` |
| Le modèle ne s'arrête pas | modèle *de base* (non Instruct), fin de tour inconnue | utilisez un modèle « Instruct » ; ajoutez `conv.options.stop` |
| Répétitions en boucle | petit modèle, température basse | `repeatPenalty` 1.15–1.3, température 0.7 |
| `ChatError: message trop long` | pièce jointe / question > contexte | augmentez `nCtx`, découpez (chapitre 7.4) |
| `ChatError: … JSON valide` | modèle trop petit ou consigne floue | `greedySampling()`, schéma concret, exemple de JSON dans le prompt système |
| `ChatError: aucun code SVG` | le modèle a répondu en texte | reformulez (« dessine… »), modèle plus grand |
| Perte `nan` à l'entraînement | taux d'apprentissage trop élevé | divisez `lr` par 3, gardez `gradClip = 1.0` |
| Perte qui ne baisse pas | `lr` trop faible, données trop variées pour la taille du modèle | augmentez `lr`, plus d'étapes, modèle plus grand |
| Mémoire saturée (LoRA) | lots trop gros | `batchSize = 1`, `gradAccum` plus grand, `seqLen` plus petit |
| Caractères « � » | coupure au milieu d'un caractère UTF-8 | utilisez `onToken`/`text` (déjà sûrs) plutôt que `tokenToPiece` |

## 13.6 Diagnostiquer

```nim
# fichier : diagnostic.nim
import std/[os, strutils]
import nimllm

let chemin = getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let g = openGguf(chemin)
echo "Architecture : ", g.getStr("general.architecture")
echo "Tokeniseur   : ", g.getStr("tokenizer.ggml.model"), " / ", g.getStr("tokenizer.ggml.pre")
echo "Gabarit      : ", (if g.has("tokenizer.chat_template"): "présent" else: "ABSENT (forcez templ)")
g.close()

let m = loadModel(chemin, verbose = true)
let conv = newChat(m, system = "Test.")
echo "Gabarit détecté : ", conv.templ
echo "Prompt envoyé :\n", renderPrompt(conv.templ, conv.messages & @[Message(role: roleUser, content: "Bonjour")])
echo "Fins de tour : "
for id in m.tokenizer.eogIds: echo "  ", id, " = ", m.tokenizer.tokens[id]

# Le modèle « comprend »-il ? Les tokens attendus doivent être en tête :
let ctx = newContext(m, 256)
let logits = ctx.eval(m.tokenizer.encode("The capital of France is", addBos = true))
for (id, p) in topTokens(logits, 3):
  echo "  ", m.tokenizer.tokenToPiece(id).escape, " ", (p*100).formatFloat(ffDecimal, 1), " %"
```

Si « Paris » n'apparaît pas en tête pour un modèle connu, le fichier est
probablement corrompu ou d'une architecture mal reconnue : signalez-le avec la
sortie de `g.describe()`.

## 13.7 Vérifier l'installation avec les tests

```sh
nim c -r tests/t_complet.nim       # test de bout en bout autonome (~30 s)
nim c -r tests/t_grad.nim          # gradients de toutes les opérations
nimble exemples                    # compile les 20 exemples
```

Tests de conformité avec llama.cpp (nécessitent une copie du dépôt llama.cpp,
indiquée par la variable `LLAMA_CPP`) :

```sh
LLAMA_CPP=~/src/llama.cpp nim c -r tests/t_tok.nim    # tokeniseurs vs jeux de tests officiels
```

**Suite :** **14 — Référence de l'API**


---

# 14 — Référence de l'API

`import nimllm` donne accès à tous les modules ci-dessous. Les signatures sont
données en notation Nim ; les paramètres avec `=` sont optionnels.

## chat — conversation

### Types

```nim
type
  Role = enum roleSystem, roleUser, roleAssistant
  Message = object
    role: Role
    content: string
  ChatTemplateKind = enum tplAuto, tplLlama3, tplChatML, tplMistral, tplLlama2, tplGemma, tplPhi3, tplRaw
  OutputFormat = enum ofText, ofMarkdown, ofJson, ofImage, ofAudio, ofFile
  StopReason = enum srEndOfText, srStopString, srMaxTokens, srContextFull, srCallback
  TokenCallback = proc (piece: string): bool {.closure.}
  GenOptions = object
    maxTokens: int; sampling: SamplerParams; stop: seq[string]; onToken: TokenCallback
  Reply = object
    text, raw: string; files: seq[string]; json: JsonNode
    promptTokens, completionTokens: int; stopReason: StopReason
    seconds, tokensPerSecond: float
  Chat = ref object
    model: LlmModel; ctx: LlmContext; system: string; history: seq[Message]
    templ: ChatTemplateKind; options: GenOptions; stripThinking: bool
    lang: string; imageSize: int; jsonRetries: int; voicePitch: float
```

### Fonctions

| Signature | Description |
|---|---|
| `newChat(m; system = ""; nCtx = 4096; templ = tplAuto; sampling = defaultSampling(); maxTokens = 512; lang = "fr"): Chat` | nouvelle conversation |
| `ask(c; prompt; attachments = @[]; format = ofText; outPath = ""; schema = ""; onToken = nil; maxTokens = 0): Reply` | pose une question |
| `askJson(c; prompt; schema = ""): JsonNode` | question à réponse JSON |
| `regenerate(c; onToken = nil): Reply` | nouveau tirage de la dernière réponse |
| `continueReply(c; maxTokens = 256; onToken = nil): Reply` | prolonge la dernière réponse |
| `add(c; role; content)` | ajoute un message sans générer |
| `reset(c)` / `undo(c)` | efface l'historique / retire le dernier échange |
| `setSampling(c; p)` | change les réglages d'échantillonnage |
| `messages(c): seq[Message]` | système + historique |
| `promptTokens(c; addGen = true): seq[int]` | prompt tokenisé |
| `saveHistory(c; path)` / `loadHistory(c; path)` / `toJson(c)` | persistance |
| `stats(c): string` | état du contexte |
| `generate(ctx; prompt: seq[int]; opts; smp = nil): Reply` | génération à partir de tokens |
| `complete(ctx; text; opts = defaultOptions()): Reply` | complétion brute |
| `defaultOptions(): GenOptions` | 512 tokens, `defaultSampling()` |
| `detectTemplate(tok): ChatTemplateKind` | gabarit d'un modèle |
| `renderPrompt(kind; msgs; addGenerationPrompt = true; bos = "<s>"): string` | texte du prompt |
| `encodeMessages(tok; kind; msgs; addGenerationPrompt = true): seq[int]` | prompt tokenisé (balises sûres) |
| `stopStringsFor(kind): seq[string]` | chaînes d'arrêt implicites |
| `extractJson(s): JsonNode` | extrait le premier JSON d'un texte |
| `stripFences(s): string` | retire les balises ``` |

## model — inférence

| Signature | Description |
|---|---|
| `loadModel(path; threads = 0; verbose = false): LlmModel` | ouvre un GGUF |
| `describe(m): string` / `paramCount(m): int` / `close(m)` | informations / fermeture |
| `m.cfg: ModelConfig` | `arch, name, dim, hidden, nLayers, nHeads, nKvHeads, headDim, vocab, ctxTrain, ropeBase, ropeDim, ropeNeox, normEps, tiedEmbeddings` |
| `m.tokenizer: Tokenizer` | tokeniseur du modèle |
| `newContext(m; nCtx = 2048; nBatch = 32): LlmContext` | cache KV |
| `eval(ctx; toks; allLogits = false): seq[float32]` | évalue des tokens |
| `evalPrompt(ctx; toks): seq[float32]` | idem avec réutilisation du préfixe en cache |
| `reset(ctx)` / `truncate(ctx; n)` / `nPast(ctx)` / `ctx.tokens` | gestion du cache |
| `memoryUsage(ctx)` / `tokensPerSecond(ctx)` | statistiques |
| `applyLora(m; path)` / `removeLora(m)` | adaptateurs à la volée |
| `embed(m; text; nCtx = 512): seq[float32]` | vecteur normalisé |
| `cosineSimilarity(a, b): float` | similarité |
| `perplexity(m; text; nCtx = 512): float` | perplexité |
| `exactMatmul: bool` (variable globale) | désactive les activations int8 |
| `newQMatrix(name; typ; rows, cols; values): QMatrix` / `matmul(w; x; y; nb)` | matrices quantifiées |

## sampler — échantillonnage

```nim
type SamplerParams = object
  temperature, topP, minP, repeatPenalty, presencePenalty, frequencyPenalty: float
  topK, repeatLastN, seed: int
  logitBias: Table[int, float]
```

| Signature | Description |
|---|---|
| `defaultSampling()` / `greedySampling()` | préréglages |
| `newSampler(p): Sampler` | échantillonneur |
| `sample(s; logits): int` / `accept(s; tok)` | tirage / mémorisation |
| `argmax(logits)` / `softmaxInPlace(x)` / `topTokens(logits; n = 5)` | utilitaires |

## tokenizer — tokens

| Signature | Description |
|---|---|
| `encode(t; text; addBos = false; parseSpecial = true): seq[int]` | texte → tokens |
| `decode(t; ids; renderSpecial = false): string` | tokens → texte |
| `tokenToPiece(t; id; renderSpecial = false): string` | un token |
| `tokenId(t; text): int` / `isSpecial(t; id)` / `vocabSize(t)` | recherche |
| `t.bosId, t.eosId, t.padId, t.unkId, t.eogIds, t.addBos, t.chatTemplate, t.tokens, t.merges, t.kind, t.pre` | champs |
| `tokenizerFromGguf(g)` / `writeToGguf(t; w)` | lecture / écriture GGUF |
| `newBpeTokenizer(tokens; merges; types; pre = ptLlama3)` / `rebuild(t)` | construction |
| `preTokenize(text; mode)` / `byteEncode(s)` / `byteDecode(s)` | outils BPE |

## tokentrain — apprentissage de tokeniseur

| Signature | Description |
|---|---|
| `trainBpe(texts; vocabSize = 2048; specials = defaultSpecials; minFreq = 2; pre = ptLlama3; verbose = false): Tokenizer` | BPE niveau octet |
| `byteLevelTokenizer(specials = defaultSpecials): Tokenizer` | un token par octet |
| `defaultSpecials` / `chatmlTemplate` | constantes |

## attachments — pièces jointes

```nim
type Attachment = object
  name, mime, text, data: string
  kind: AttachmentKind      # akText, akImage, akPdf, akAudio, akBinary
  width, height: int
  image: Image              # pixels si PNG/BMP/PPM
```

| Signature | Description |
|---|---|
| `attach(path): Attachment` | depuis un fichier |
| `attachText(name; content)` / `attachData(name; data)` | depuis la mémoire |
| `toPrompt(a; maxChars = 12000): string` | texte inséré dans le message |
| `decodePng(data): Image` / `pdfText(data): string` / `describeImage(img)` | décodeurs |

## image — images

| Signature | Description |
|---|---|
| `rgb(r, g, b; a = 1.0): Color` / `parseColor(s)` | couleurs |
| `newImage(w, h; bg)` / `getPixel` / `setPixel` | image RVB |
| `writeBmp` / `writePpm` / `save` / `readBmp` / `readPpm` | fichiers |
| `newCanvas(w, h; bg; ss = 3): Canvas` / `finish(c): Image` | dessin anticrénelé |
| `fillRect`, `fillCircle`, `fillPolygon`, `fillPolygons`, `strokePolyline`, `drawLine`, `ellipsePoints` | primitives |
| `renderSvg(svg; width = 0; height = 0; bg): Image` | rendu SVG |
| `extractSvg(text)` / `svgSize(svg)` / `parsePath(d)` | outils SVG |

## audio — sons

| Signature | Description |
|---|---|
| `speak(text; lang = "fr"; pitch = 115.0; speed = 1.0; sampleRate = 16000): Audio` | synthèse vocale |
| `renderMelody(notation; bpm = 120.0; sampleRate = 22050): Audio` | mélodie |
| `writeWav(a; path)` / `readWav(path)` / `readWavInfo(path)` | fichiers WAV |
| `concat(a; b)` / `silence(a; secondes)` / `normalize(a)` / `duration(a)` | montage |
| `textToPhonemes(text; lang)` / `numberToFrench(n)` / `numberToEnglish(n)` / `noteFrequency(note)` | outils |

## autograd — tenseurs différentiables

| Signature | Description |
|---|---|
| `newTensor(shape; requiresGrad = false)`, `fromSeq(data; shape)`, `full(shape; v)`, `randn(shape; std; rng)`, `param(shape; std; rng; name)`, `scalar(v)` | création |
| `t.data`, `t.grad`, `t.shape`, `rows`, `cols`, `numel`, `item`, `reshape`, `detach` | accès |
| `+`, `-`, `*`, `scale`, `sum`, `mean`, `linear(x, w, b = nil)`, `matmul`, `concatCols` | opérations |
| `relu`, `silu`, `gelu`, `sigmoid`, `tanhT`, `softmax`, `rmsnorm`, `dropout` | fonctions |
| `embedding`, `rope`, `causalAttention`, `crossEntropy`, `mseLoss` | blocs de LLM |
| `backward(loss)` / `zeroGrad(t)` / `noGrad: …` | gradients |
| `needsGrad(…)` / `makeNode(res; parents; backward)` / `ensureGrad(t)` | opérations personnalisées |
| `gradCheck(f; t; eps = 1e-2; samples = 10): float` | vérification |

## nn — modèles entraînables

| Signature | Description |
|---|---|
| `newModelConfig(vocab; dim = 256; layers = 4; heads = 4; kvHeads = 0; hidden = 0; ctx = 256; ropeBase = 10000.0; tied = true; name): ModelConfig` | architecture |
| `newTransformer(cfg; tok; seed = 42): Transformer` | modèle neuf |
| `loadForTraining(path; mode = tmLora; lora = defaultLora(); seed = 42; threads = 0): Transformer` | depuis un GGUF |
| `defaultLora(): LoraConfig` (`rank`, `alpha`, `targets`) | réglages LoRA |
| `parameters(t)` / `parameterCount(t)` | paramètres |
| `forward(t; ids; B, T)` / `hidden(t; ids; B, T)` / `loss(t; inputs; targets; B, T)` | calcul |
| `generateText(t; prompt; maxTokens = 50; temperature = 0.8)` | génération de contrôle |
| `saveGguf(t; path; wtype = gtF32)` / `saveLora(t; path)` | export |
| `mergeLora(base; adapter; out; outType = gtQ8_0)` / `quantizeModel(in; out; wtype; keepOutput = true)` | fichiers |
| `qlinear(x; q)` / `qembedding(q; ids)` / `Linear.forward(x)` / `addLora(l; rank; alpha; rng)` | couches |

## train — entraînement

| Signature | Description |
|---|---|
| `newAdamW(params; lr = 3e-4; beta1 = 0.9; beta2 = 0.95; eps = 1e-8; weightDecay = 0.1)` / `newSGD(params; lr; momentum; weightDecay)` | optimiseurs |
| `zeroGrad(o)` / `update(o)` / `saveState(o; path)` / `loadState(o; path)` / `o.lr`, `o.step` | optimiseur |
| `clipGradNorm(params; maxNorm)` / `gradNorm(params)` / `cosineLr(step, warmup, total, maxLr, minLr)` | outils |
| `newTextDataset(tok; text)` / `newTextDatasetFromFiles(tok; paths)` | données texte |
| `newChatDataset(tok; templ; conversations)` / `loadChatJsonl(path; tok; templ; system = "")` / `addChatExample` | dialogues |
| `split(d; valFraction = 0.1)` / `getBatch(d; B, T): Batch` / `len(d)` | manipulation |
| `defaultTrainConfig(): TrainConfig` | `steps, batchSize, seqLen, gradAccum, lr, minLr, warmup, weightDecay, gradClip, evalEvery, evalBatches, logEvery, saveEvery, savePath, sampleEvery, samplePrompt, seed` |
| `train(t; data; cfg; valData = nil; opt = nil; onLog = nil): Optimizer` | boucle complète |
| `evaluate(t; d; batches = 4; B = 8; T = 64): float` | perte moyenne |
| `saveCheckpoint(t; opt; prefix)` / `printLog(l)` | sauvegarde / journal |

## gguf et quant — fichiers et formats

| Signature | Description |
|---|---|
| `openGguf(path): GgufFile` / `close(g)` / `describe(g)` | lecture |
| `g.kv`, `g.tensors`, `has`, `getInt`, `getFloat`, `getStr`, `getBool`, `getStrArray`, `getIntArray`, `getFloatArray` | métadonnées |
| `tensorF32(g; name)`, `GgufTensorInfo.dims/typ/data/numElements/byteSize` | tenseurs |
| `newGgufWriter()`, `setKV`, `addTensor`, `addTensorF32(name; dims; values; typ)`, `addTensorRaw`, `copyMetadata`, `write` | écriture |
| `gStr`, `gU32`, `gI32`, `gU64`, `gF32`, `gBool`, `gArrStr`, `gArrI32`, `gArrF32` | valeurs |
| `GgmlType` (`gtF32`, `gtF16`, `gtBF16`, `gtQ8_0`, `gtQ4_0`… `gtQ6_K`), `parseGgmlType("q4_k")` | types |
| `quantize(t; data)`, `quantizeRow`, `dequantRow`, `blockSize`, `typeSize`, `rowBytes`, `bitsPerWeight` | conversions |
| `floatToHalf` / `halfToFloat` / `floatToBf16` / `bf16ToFloat` | demi-précision |

## parallel — threads

| Signature | Description |
|---|---|
| `setThreads(n = 0)` / `numThreads()` / `shutdownPool()` | pool |
| `parallelFor(n; fn: TaskFn; ctx: pointer; minChunk = 1)` | boucle parallèle (`fn(ctx, first, last, worker)`) |

## inflate — décompression

| Signature | Description |
|---|---|
| `zlibDecompress(data)` / `inflateRaw(data)` | flux zlib / DEFLATE |
