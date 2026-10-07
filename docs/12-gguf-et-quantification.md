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

**Suite :** [13 — Performances et dépannage](13-performances-et-depannage.md)
