# Vérifie la tokenisation contre les fichiers de test de llama.cpp.
import ../src/nimllm/[gguf, tokenizer]
import std/[strutils, os]
proc check(name: string) =
  let base = getEnv("LLAMA_CPP", "../llama.cpp") / "models/ggml-vocab-" & name & ".gguf"
  let g = openGguf(base)
  let t = tokenizerFromGguf(g)
  let inp = readFile(base & ".inp").split("\n__ggml_vocab_test__\n")
  let outp = readFile(base & ".out").split("\n")
  var ok, bad = 0
  for i, s in inp:
    if i >= outp.len: break
    var exp: seq[int]
    for x in outp[i].splitWhitespace: exp.add parseInt(x)
    let got = t.encode(s, addBos = false, parseSpecial = false)
    if got == exp: inc ok
    else:
      inc bad
      if bad <= 4: echo "  ÉCHEC ", s.escape, "\n   attendu ", exp, "\n   obtenu  ", got
    # aller-retour
    let dec = t.decode(got)
    if dec != s and t.kind == tkBpe: echo "  décodage différent: ", s.escape, " -> ", dec.escape
  echo name, ": ", ok, " ok / ", bad, " échecs"
let noms = if paramCount() > 0: commandLineParams()
           else: @["llama-bpe", "llama-spm", "qwen2", "gpt-2", "phi-3", "deepseek-llm"]
for n in noms: check(n)
