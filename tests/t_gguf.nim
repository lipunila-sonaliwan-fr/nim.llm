import ../src/nimllm/[gguf, quant]
import std/[random, math, os]
let dir = getEnv("LLAMA_CPP", "../llama.cpp")
if fileExists(dir / "models/ggml-vocab-llama-bpe.gguf"):
  let g = openGguf(dir / "models/ggml-vocab-llama-bpe.gguf")
  echo g.describe(5)
# roundtrip quantization
randomize(1)
var x = newSeq[float32](512)
for v in x.mitems: v = float32(gauss())
for t in [gtF16, gtBF16, gtQ8_0, gtQ4_0, gtQ4_1, gtQ5_0, gtQ5_1, gtQ4_K, gtQ5_K, gtQ6_K]:
  let q = quantize(t, x)
  let y = dequantRow(t, unsafeAddr q[0], x.len)
  var e = 0.0
  for i in 0..<x.len: e += (x[i]-y[i])^2
  echo t, " rmse=", sqrt(e/x.len.float)
