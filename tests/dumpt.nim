import ../src/nimllm/gguf
import std/os
let g = openGguf(paramStr(1))
let v = g.tensorF32(paramStr(2))
let f = open(paramStr(3), fmWrite)
discard f.writeBuffer(unsafeAddr v[0], v.len*4)
f.close()
