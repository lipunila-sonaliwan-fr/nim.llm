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

**Suite :** [03 — Conversation et contexte](03-conversation-et-contexte.md)
