import ../src/nimllm/[autograd, parallel]
import std/[random, strutils]
setThreads(2)
var rng = initRand(3)
let x = randn([6, 8], 1.0, rng)
let w = randn([5, 8], 0.5, rng)
let w2 = randn([8, 5], 0.5, rng)
let g = full([8], 1.0, true)
for v in g.data.mitems: v = float32(1 + rng.gauss()*0.1)
let tgt = @[1, 4, -1, 0, 2, 3]
template chk(name: string; body: untyped) =
  let f = proc (): Tensor = body
  echo name, ": x=", gradCheck(f, x).formatFloat(ffScientific, 2)
chk "linear+CE": crossEntropy(linear(x, w), tgt)
chk "matmul": sum(matmul(x, w2) * matmul(x, w2))
chk "silu/gelu": sum(silu(x) * gelu(x))
chk "rmsnorm": sum(rmsnorm(x, g) * rmsnorm(x, g).scale(0.5) + x)
chk "softmax": sum(softmax(x) * x)
echo "linear w: ", gradCheck(proc (): Tensor = crossEntropy(linear(x, w), tgt), w)
echo "rmsnorm g: ", gradCheck(proc (): Tensor = sum(rmsnorm(x, g) * x), g)
let invf = @[1.0'f32, 0.3, 0.1, 0.01]
let pos = @[0, 1, 2, 0, 1, 2]
chk "rope": sum(rope(x, 1, 8, pos, invf) * x)
chk "rope neox": sum(rope(x, 2, 4, pos, invf[0..1], true) * x.scale(0.3))
# attention : B=2, T=3, H=2, Hkv=1, D=4 -> q [6, 8], k/v [6, 4]
let k = randn([6, 4], 1.0, rng)
let v = randn([6, 4], 1.0, rng)
let att = proc (): Tensor = sum(causalAttention(x, k, v, 2, 3, 2, 1, 4) * x)
echo "attention q: ", gradCheck(att, x), " k: ", gradCheck(att, k), " v: ", gradCheck(att, v)
let ids = @[1, 3, 3, 0]
echo "embedding: ", gradCheck(proc (): Tensor = sum(embedding(w, ids) * embedding(w, ids)), w)
echo "attention v (eps 1e-2, 40): ", gradCheck(att, v, 1e-2, 40)
echo "attention k (eps 1e-2, 40): ", gradCheck(att, k, 1e-2, 40)
