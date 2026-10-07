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

**Suite :** [05 — Pièces jointes](05-pieces-jointes.md)
