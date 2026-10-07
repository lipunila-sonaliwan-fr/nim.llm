# 11 — Créer un nouveau modèle

Objectif : concevoir et entraîner un modèle de langage **à partir de zéro**, puis
l'utiliser exactement comme Llama 3.2 (et même dans llama.cpp ou Ollama).

Le programme complet de ce chapitre est
[`examples/ex17_creer_modele.nim`](../examples/ex17_creer_modele.nim) : en deux
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
  ([`examples/ex20_reprise_entrainement.nim`](../examples/ex20_reprise_entrainement.nim)).
* **Distillation** : générez des dialogues avec un grand modèle (chapitre 7.6)
  pour entraîner un petit modèle spécialisé.

**Suite :** [12 — GGUF et quantification](12-gguf-et-quantification.md)
