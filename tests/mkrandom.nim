# Crée un petit modèle aléatoire (vocabulaire réel) pour tester l'inférence.
import ../src/nimllm/[gguf, quant]
import std/[random, os, strutils, math]
let vocabFile = paramStr(1)
let outPath = paramStr(2)
let wt = parseGgmlType(paramStr(3))
let arch = if paramCount() >= 4: paramStr(4) else: "llama"
let v = openGguf(vocabFile)
let w = newGgufWriter()
for k, val in v.kv:
  if k.startsWith("tokenizer.") or k == "general.name": w.setKV(k, val)
let nvocab = v.getStrArray("tokenizer.ggml.tokens").len
let dim = 256; let ffn = 512; let nl = 2; let nh = 4; let nkv = 2; let hd = dim div nh
w.setKV("general.architecture", gStr(arch))
w.setKV(arch & ".context_length", gU32(512))
w.setKV(arch & ".embedding_length", gU32(dim))
w.setKV(arch & ".feed_forward_length", gU32(ffn))
w.setKV(arch & ".block_count", gU32(nl))
w.setKV(arch & ".attention.head_count", gU32(nh))
w.setKV(arch & ".attention.head_count_kv", gU32(nkv))
w.setKV(arch & ".rope.freq_base", gF32(500000.0))
w.setKV(arch & ".attention.layer_norm_rms_epsilon", gF32(1e-5))
w.setKV(arch & ".rope.dimension_count", gU32(hd))
w.setKV(arch & ".vocab_size", gU32(nvocab))
randomize(42)
proc rnd(n: int; s: float): seq[float32] =
  result = newSeq[float32](n)
  for x in result.mitems: x = float32(gauss() * s)
proc ones(n: int): seq[float32] =
  result = newSeq[float32](n)
  for x in result.mitems: x = float32(1.0 + gauss()*0.1)
w.addTensorF32("token_embd.weight", @[dim, nvocab], rnd(dim*nvocab, 1.0), wt)
w.addTensorF32("output_norm.weight", @[dim], ones(dim))
w.addTensorF32("output.weight", @[dim, nvocab], rnd(dim*nvocab, 0.08), wt)
if arch == "llama":
  var f = newSeq[float32](hd div 2)
  for i in 0..<f.len: f[i] = float32(1.0 + i.float / 8.0)
  w.addTensorF32("rope_freqs.weight", @[hd div 2], f)
for l in 0..<nl:
  let p = "blk." & $l & "."
  w.addTensorF32(p & "attn_norm.weight", @[dim], ones(dim))
  w.addTensorF32(p & "ffn_norm.weight", @[dim], ones(dim))
  w.addTensorF32(p & "attn_q.weight", @[dim, dim], rnd(dim*dim, 0.08), wt)
  w.addTensorF32(p & "attn_k.weight", @[dim, nkv*hd], rnd(dim*nkv*hd, 0.08), wt)
  w.addTensorF32(p & "attn_v.weight", @[dim, nkv*hd], rnd(dim*nkv*hd, 0.08), wt)
  w.addTensorF32(p & "attn_output.weight", @[dim, dim], rnd(dim*dim, 0.08), wt)
  w.addTensorF32(p & "ffn_gate.weight", @[dim, ffn], rnd(dim*ffn, 0.08), wt)
  w.addTensorF32(p & "ffn_up.weight", @[dim, ffn], rnd(dim*ffn, 0.08), wt)
  w.addTensorF32(p & "ffn_down.weight", @[ffn, dim], rnd(dim*ffn, 0.06), wt)
  if arch == "qwen2":
    w.addTensorF32(p & "attn_q.bias", @[dim], rnd(dim, 0.1))
    w.addTensorF32(p & "attn_k.bias", @[nkv*hd], rnd(nkv*hd, 0.1))
    w.addTensorF32(p & "attn_v.bias", @[nkv*hd], rnd(nkv*hd, 0.1))
w.write(outPath)
echo "écrit ", outPath
