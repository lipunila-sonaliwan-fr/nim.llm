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

**Suite :** [02 — Premiers pas](02-premiers-pas.md)
