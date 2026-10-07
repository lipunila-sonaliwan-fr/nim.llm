# Paquet nimllm
version       = "1.0.0"
author        = "nimllm"
description   = "Bibliothèque LLM 100 % Nim : inférence GGUF (Llama 3.x, Mistral, Qwen), dialogue, pièces jointes, sorties texte/JSON/image/audio, entraînement (LoRA, création de modèles)."
license       = "MIT"
srcDir        = "src"

requires "nim >= 2.0.0"

task test, "Exécute les tests autonomes":
  exec "nim c -r tests/t_complet.nim"
  exec "nim c -r tests/t_grad.nim"

task exemples, "Compile tous les exemples":
  for f in listFiles("examples"):
    if f.endsWith(".nim"):
      exec "nim c -d:release --hints:off " & f
