# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm - bibliothèque LLM 100 % Nim, sans dépendance externe.
#
#  ___    _              ____
# | \ \  (_)_ __ ___    / / /\/\
# | |\ \ | | '_ ` _ \  / / /    \
# | | \ \| | | | | | |/ / / /\/\ \
# |_|  \_\_|_| |_| |_/_/_/\/    \/1.0.0#
#
# Importer `nimllm` donne accès à l'ensemble des modules :
#
# Module          Rôle
# gguf          : lecture/écriture des fichiers GGUF (format llama.cpp)
# quant         : formats de poids quantifiés (Q4_K, Q8_0, F16...)
# tokenizer     : tokeniseurs BPE (Llama 3, Qwen) et SentencePiece (Llama 2, Mistral)
# model         : inférence Transformer (llama, mistral, qwen2, qwen3)
# sampler       : échantillonnage (température, top-k, top-p, min-p, pénalités)
# chat          : conversation, pièces jointes, formats de réponse
# attachments   : conversion des pièces jointes (texte, PDF, images, audio)
# image         : images BMP/PPM, dessin, rendu SVG
# audio         : WAV, synthèse vocale par formants, mélodies
# autograd      : tenseurs avec différentiation automatique
# nn            : couches, Transformer entraînable, LoRA
# train         : optimiseurs, boucle d'entraînement, données, export GGUF
# tokentrain    : entraînement d'un tokeniseur BPE
# parallel      : pool de threads
#

import nimllm/[parallel, quant, gguf, tokenizer, model, sampler, chat,
               attachments, image, audio, inflate, autograd, nn, train, tokentrain]
export parallel, quant, gguf, tokenizer, model, sampler, chat,
       attachments, image, audio, inflate, autograd, nn, train, tokentrain

const nimllmVersion* = "1.0.0"
