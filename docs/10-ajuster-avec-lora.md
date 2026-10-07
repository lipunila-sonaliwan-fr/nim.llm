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

Exemple fourni : [`examples/donnees/faq_boulangerie.jsonl`](../examples/donnees/faq_boulangerie.jsonl).

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

Version commentée dans les exemples : [`examples/ex18_lora.nim`](../examples/ex18_lora.nim).

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

**Suite :** [11 — Créer un nouveau modèle](11-creer-un-modele.md)
