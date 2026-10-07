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

**Suite :** [14 — Référence de l'API](14-reference-api.md)
