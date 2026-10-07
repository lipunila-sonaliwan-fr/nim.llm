# Modèle aléatoire aux dimensions de Llama 3.2 1B (banc d'essai de vitesse).
import ../src/nimllm/[gguf, quant]
import std/[random, os, strutils]
let v = openGguf(getEnv("LLAMA_CPP", "../llama.cpp") / "models/ggml-vocab-llama-bpe.gguf")
let w = newGgufWriter()
for k, val in v.kv:
  if k.startsWith("tokenizer."): w.setKV(k, val)
let dim = 2048; let ffn = 8192; let nl = parseInt(paramStr(2)); let nh = 32; let nkv = 8; let hd = 64
let nv = 128256
let arch = "llama"
w.setKV("general.architecture", gStr(arch)); w.setKV("general.name", gStr("bench-1b"))
w.setKV(arch & ".context_length", gU32(8192)); w.setKV(arch & ".embedding_length", gU32(dim))
w.setKV(arch & ".feed_forward_length", gU32(ffn)); w.setKV(arch & ".block_count", gU32(nl))
w.setKV(arch & ".attention.head_count", gU32(nh)); w.setKV(arch & ".attention.head_count_kv", gU32(nkv))
w.setKV(arch & ".rope.freq_base", gF32(500000.0)); w.setKV(arch & ".attention.layer_norm_rms_epsilon", gF32(1e-5))
w.setKV(arch & ".rope.dimension_count", gU32(hd)); w.setKV(arch & ".vocab_size", gU32(nv))
var rng = initRand(1)
proc rnd(n: int; s: float): seq[float32] =
  result = newSeq[float32](n)
  for x in result.mitems: x = float32((rng.rand(2.0) - 1.0) * s)
proc ones(n: int): seq[float32] =
  result = newSeq[float32](n)
  for x in result.mitems: x = 1
w.addTensorF32("token_embd.weight", @[dim, nv], rnd(dim*nv, 0.05), gtQ6_K)
w.addTensorF32("output_norm.weight", @[dim], ones(dim))
for l in 0..<nl:
  let p = "blk." & $l & "."
  w.addTensorF32(p & "attn_norm.weight", @[dim], ones(dim))
  w.addTensorF32(p & "ffn_norm.weight", @[dim], ones(dim))
  w.addTensorF32(p & "attn_q.weight", @[dim, dim], rnd(dim*dim, 0.03), gtQ4_K)
  w.addTensorF32(p & "attn_k.weight", @[dim, nkv*hd], rnd(dim*nkv*hd, 0.03), gtQ4_K)
  w.addTensorF32(p & "attn_v.weight", @[dim, nkv*hd], rnd(dim*nkv*hd, 0.03), gtQ6_K)
  w.addTensorF32(p & "attn_output.weight", @[dim, dim], rnd(dim*dim, 0.03), gtQ4_K)
  w.addTensorF32(p & "ffn_gate.weight", @[dim, ffn], rnd(dim*ffn, 0.03), gtQ4_K)
  w.addTensorF32(p & "ffn_up.weight", @[dim, ffn], rnd(dim*ffn, 0.03), gtQ4_K)
  w.addTensorF32(p & "ffn_down.weight", @[ffn, dim], rnd(dim*ffn, 0.02), gtQ6_K)
w.write(paramStr(1))
