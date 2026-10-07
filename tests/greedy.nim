# Génération gloutonne : affiche les identifiants produits.
import ../src/nimllm/[model, tokenizer]
import std/[os, strutils]
let m = loadModel(paramStr(1))
let ctx = newContext(m, 512)
let prompt = paramStr(2)
let n = parseInt(paramStr(3))
var toks = m.tokenizer.encode(prompt, addBos = m.tokenizer.addBos)
var logits = ctx.eval(toks)
var outIds: seq[int]
for i in 0..<n:
  var best = 0
  for k in 1..<logits.len:
    if logits[k] > logits[best]: best = k
  outIds.add best
  logits = ctx.eval([best])
echo "prompt: ", toks
echo "ids: ", outIds
echo "texte: ", m.tokenizer.decode(outIds).escape
