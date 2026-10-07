# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 15 — Les bases de l'apprentissage : tenseurs, gradients et
# optimiseur, sur deux petits problèmes (régression linéaire et XOR).
#
#   nim c -r examples/ex15_autograd.nim
#
# (aucun modèle de langage nécessaire)

import std/[random, strutils]
import nimllm

var rng = initRand(2024)

# 1. Un gradient calculé à la main.
# f(x) = somme(x²)  => df/dx = 2x
let x = fromSeq(@[1.0'f32, -2.0, 3.0], [3], requiresGrad = true)
let f = sum(x * x)
backward(f)
echo "f = ", f.item, "   df/dx = ", x.grad                        # @[2.0, -4.0, 6.0].

# 2. Régression linéaire : retrouver y = 3x + 1
var xs, ys: seq[float32]
for i in 0 ..< 64:
  let v = rng.rand(2.0) - 1.0
  xs.add float32(v)
  ys.add float32(3*v + 1 + rng.gauss()*0.05)
let X = fromSeq(xs, [64, 1])
let Y = fromSeq(ys, [64, 1])
let w = param([1, 1], 0.1, rng, "w")                              # poids [sorties, entrées].
let b = full([1], 0, true, "b")                                   # biais.
let opt = newSGD(@[w, b], lr = 0.1, momentum = 0.9)
for etape in 1 .. 100:
  opt.zeroGrad()
  let perte = mseLoss(linear(X, w, b), Y)
  backward(perte)
  opt.update()
  if etape mod 25 == 0:
    echo "étape ", etape, " perte=", perte.item.formatFloat(ffDecimal, 5),
         "  w=", w.data[0].formatFloat(ffDecimal, 3), " b=", b.data[0].formatFloat(ffDecimal, 3)

# 3. Réseau de neurones à 2 couches : le XOR.
# Entrées 2 → 16 neurones cachés (tanh) → 2 classes (entropie croisée).
let entrees = fromSeq(@[0'f32, 0, 0, 1, 1, 0, 1, 1], [4, 2])
let cibles = @[0, 1, 1, 0]
let w1 = param([16, 2], 1.0, rng, "w1")
let b1 = full([16], 0, true, "b1")
let w2 = param([2, 16], 0.5, rng, "w2")
let b2 = full([2], 0, true, "b2")
let adam = newAdamW(@[w1, b1, w2, b2], lr = 0.05, weightDecay = 0)

proc predire(e: Tensor): Tensor = linear(tanhT(linear(e, w1, b1)), w2, b2)

for etape in 1 .. 300:
  adam.zeroGrad()
  let perte = crossEntropy(predire(entrees), cibles)
  backward(perte)
  adam.update()
  if etape mod 100 == 0: echo "XOR étape ", etape, " perte=", perte.item.formatFloat(ffDecimal, 4)

noGrad:                          # pas de graphe en inférence.
  let probas = softmax(predire(entrees))
  for i in 0 ..< 4:
    let p1 = probas.data[2*i + 1]
    echo "XOR(", entrees.data[2*i].int, ", ", entrees.data[2*i+1].int, ") = ",
         (if p1 > 0.5: 1 else: 0), "   (probabilité de 1 : ", p1.formatFloat(ffDecimal, 3), ")"

# 4. Vérifier une dérivée par différences finies.
let t = randn([3, 4], 1.0, rng)
let ecart = gradCheck(proc (): Tensor = sum(silu(t) * t), t)
echo "Écart relatif gradient analytique / numérique : ", ecart.formatFloat(ffScientific, 2)

# 5. Opération personnalisée avec sa propre dérivée.
proc carre(a: Tensor): Tensor =
  # y = a², dy/da = 2a
  result = newTensor(a.shape)
  for i in 0 ..< a.numel: result.data[i] = a.data[i] * a.data[i]
  if needsGrad(a):
    let r = result
    result.makeNode(@[a], proc () =
      a.ensureGrad()
      for i in 0 ..< a.numel: a.grad[i] += r.grad[i] * 2 * a.data[i])

echo "Vérification de l'opération personnalisée : ",
     gradCheck(proc (): Tensor = sum(carre(t)), t).formatFloat(ffScientific, 2)
