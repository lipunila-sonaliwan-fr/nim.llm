# Tests

| Fichier | Rôle | Prérequis |
|---|---|---|
| `t_complet.nim` | bout en bout : tokeniseur, entraînement, export GGUF, quantification, dialogue, image, audio, JSON | aucun |
| `t_grad.nim` | vérification par différences finies de toutes les opérations d'autograd | aucun |
| `t_gguf.nim` | lecture GGUF, aller-retour de quantification | (optionnel) `LLAMA_CPP` |
| `t_tok.nim` | tokeniseurs comparés aux jeux de tests officiels de llama.cpp | `LLAMA_CPP` = dépôt llama.cpp |
| `mkrandom.nim` | crée un petit modèle aléatoire avec un vocabulaire réel (comparaisons avec llama.cpp) | `LLAMA_CPP` |
| `mkbig.nim` | crée un modèle aléatoire aux dimensions de Llama 3.2 1B (banc d'essai) | `LLAMA_CPP` |
| `greedy.nim` | génération gloutonne (identifiants), pour comparer avec `llama-simple` | un modèle GGUF |
| `dumpt.nim` | écrit un tenseur déquantifié en binaire (comparaison avec gguf-py) | un modèle GGUF |

```sh
nim c -r tests/t_complet.nim
LLAMA_CPP=~/src/llama.cpp nim c -r tests/t_tok.nim
```
