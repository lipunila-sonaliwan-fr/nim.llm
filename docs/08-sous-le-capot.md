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

**Suite :** [09 — Bases de l'apprentissage](09-apprentissage-bases.md)
