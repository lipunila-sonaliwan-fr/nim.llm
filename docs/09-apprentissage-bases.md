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

**Suite :** [10 — Ajuster un modèle avec LoRA](10-ajuster-avec-lora.md)
