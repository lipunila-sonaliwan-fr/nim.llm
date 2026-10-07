<div align="center">

[![](nimllm.png)]()

# nimllm

**Run, fine-tune and build language models in pure Nim.**
No dependencies: no llama.cpp, no Python, no BLAS, no C library.

[![Nim](https://img.shields.io/badge/Nim-%E2%89%A5%202.0-ffe953?logo=nim&logoColor=black)](https://nim-lang.org)
[![Dependencies](https://img.shields.io/badge/dependencies-none-2ea44f)](#installation)
[![Format](https://img.shields.io/badge/models-GGUF-6f42c1)](docs/12-gguf-et-quantification.md)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey)](LICENCE.md)
[![Docs](https://img.shields.io/badge/docs-French-0366d6)](nimllm-doc-fr.md)
[![Docs](https://img.shields.io/badge/docs-English-0366d6)](nimllm-doc-en.md)

**English** · [Français](LISEZMOI.md)

[Quick start](#quick-start) ·
[Features](#features) ·
[Documentation](#documentation) ·
[Examples](#examples) ·
[Limitations](#limitations) ·
[License](#license)

</div>

---

```nim
import nimllm

let model = loadModel("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let chat = newChat(model, system = "You are a concise assistant.")
echo chat.ask("What is the capital of Japan?").text
# The capital of Japan is Tokyo.
```

nimllm reads **GGUF** files (the format used by llama.cpp, Ollama and LM Studio)
and provides everything you need to:

- **chat** with a model: system prompt, history, attachments, answers as text,
  JSON, image or audio;
- **fine-tune** an existing model such as Llama 3.2 on your own data with LoRA;
- **build** a new model from scratch: tokenizer, architecture, training, and
  export to a GGUF file that runs anywhere.

## Quick start

### Installation

You need [Nim 2.0 or later](https://nim-lang.org/install.html) and a C compiler.

```sh
git clone https://github.com/<your-account>/nimllm.git
cd nimllm
nimble install          # or, without installing: nim c --path:src ...
```

### Check that everything works (no model needed)

```sh
nim c -r tests/t_complet.nim
```

This test builds a tokenizer and trains a small model. It exports the model to
GGUF, quantizes it and chats with it. It also checks the image, audio and JSON
outputs. It takes about 30 seconds.

### Use a real model

Download a GGUF model, for example **Llama 3.2 1B Instruct Q4_K_M** (~0.8 GB)
from Hugging Face. If you already use Ollama, its models are GGUF files too (see
[installation](docs/01-installation.md#13-obtenir-un-modèle), in French).

```sh
export NIMLLM_MODELE=~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

nim c -r examples/ex01_bonjour.nim          # one question, one answer
nim c -r examples/ex04_chat_terminal.nim    # full interactive assistant
nim c -r examples/ex17_creer_modele.nim     # build your own model (~2 min, no download)
```

> [!IMPORTANT]
> Always compile with `-d:release`. The `config.nims` in `examples/` does it for
> you. In debug mode, computation is 10 to 30 times slower.

## Features

| Area | What is supported |
|---|---|
| **Models** | Memory-mapped GGUF v2/v3. Architectures `llama` (Llama 2, 3, 3.1, 3.2, Mistral, TinyLlama…), `qwen2`, `qwen3`. Weights F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K to Q6_K |
| **Tokenizers** | Byte-level BPE (Llama 3, Qwen, GPT-2) and SentencePiece (Llama 2, Mistral) |
| **Chat** | System prompt, multi-turn history, Llama 3, ChatML, Mistral, Llama 2, Gemma and Phi-3 templates. KV cache reuse, context overflow handling, save and resume, undo, regenerate, continue |
| **Generation** | Token streaming, temperature, top-k, top-p, min-p, penalties, seed, stop strings, logit bias |
| **Attachments** | Text, code, CSV, JSON, Markdown, HTML, PDF (text extraction), PNG, BMP, PPM (decoded), JPEG, GIF, WebP (dimensions), WAV |
| **Output formats** | Text, Markdown, JSON (validated, with retries), **image** (SVG rasterized to BMP), **audio** (WAV via speech synthesis), **file** |
| **Tools** | Embeddings, semantic search (RAG), perplexity, function calling (agents) |
| **Training** | Tensors with automatic differentiation, AdamW, SGD, cosine schedule, gradient clipping and accumulation, checkpoints |
| **Fine-tuning** | LoRA on a quantized model, applied on the fly or merged into a new GGUF |
| **Model creation** | BPE tokenizer, configurable Llama-style Transformer, pre-training, chat fine-tuning, GGUF export, quantization |

<details>
<summary><b>A few API examples</b></summary>

**Streaming answer with an attachment**

```nim
let r = chat.ask("Which item is over budget?",
                 attachments = @[attach("budget.csv")],
                 onToken = proc (s: string): bool = (stdout.write s; true))
```

**Structured JSON answer**

```nim
let ad = chat.askJson("Extract the details of this listing: ...",
  schema = """{"item": str, "price_eur": int, "city": str}""")
echo ad["price_eur"].getInt
```

**Image and audio**

```nim
discard chat.ask("Draw a house and a sun.", format = ofImage, outPath = "house.svg")   # .svg + .bmp
discard chat.ask("Wish me a nice day.", format = ofAudio, outPath = "hello.wav")
```

**LoRA fine-tuning of Llama 3.2**

```nim
let m = loadForTraining("Llama-3.2-1B-Instruct-Q4_K_M.gguf", tmLora, defaultLora())
let data = loadChatJsonl("faq.jsonl", m.tokenizer, detectTemplate(m.tokenizer))
discard m.train(data, defaultTrainConfig())
m.saveLora("faq.lora.gguf")
mergeLora("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "faq.lora.gguf", "llama-faq.gguf")
```

**New model from scratch**

```nim
let tok = trainBpe([corpus], vocabSize = 2000)
let m = newTransformer(newModelConfig(tok.vocabSize, dim = 256, layers = 6, heads = 8), tok)
discard m.train(newTextDataset(tok, corpus), defaultTrainConfig())
m.saveGguf("my-model.gguf")      # usable with nimllm, llama.cpp, Ollama…
```

</details>

## Documentation

> [!NOTE]
> The documentation is currently written in **French**. The API itself (function
> and type names) is in English.

The documentation is **progressive**: each chapter builds on the previous one,
from a first "hello" to creating a model. Every chapter contains complete
programs, ready to compile.

| | Chapter | Content |
|---|---|---|
| 🟢 | [01 Installation](docs/01-installation.md) | Nim, compiling, choosing a model, memory |
| 🟢 | [02 First steps](docs/02-premiers-pas.md) | Asking a question, streaming, errors |
| 🟢 | [03 Conversation and context](docs/03-conversation-et-contexte.md) | System prompt, history, saving, context window |
| 🟡 | [04 Generation settings](docs/04-reglages-generation.md) | Temperature, top-k/p, repetition, reproducibility |
| 🟡 | [05 Attachments](docs/05-pieces-jointes.md) | Text, CSV, PDF, images, sound |
| 🟡 | [06 Output formats](docs/06-formats-de-reponse.md) | JSON, image, audio, file |
| 🟡 | [07 Practical use cases](docs/07-cas-pratiques.md) | RAG, summarizing long documents, agents, HTTP server |
| 🟠 | [08 Under the hood](docs/08-sous-le-capot.md) | Tokens, logits, KV cache, templates, embeddings |
| 🟠 | [09 Machine learning basics](docs/09-apprentissage-bases.md) | Tensors, gradients, optimizers |
| 🔴 | [10 Fine-tuning with LoRA](docs/10-ajuster-avec-lora.md) | Specializing Llama 3.2 on your data |
| 🔴 | [11 Building a model](docs/11-creer-un-modele.md) | Tokenizer, architecture, pre-training, export |
| 🔴 | [12 GGUF and quantization](docs/12-gguf-et-quantification.md) | Inspecting, writing and quantizing models |
| ⚙️ | [13 Performance and troubleshooting](docs/13-performances-et-depannage.md) | Speed, threads, common errors |
| 📖 | [14 API reference](docs/14-reference-api.md) | Every public function |

🟢 beginner · 🟡 intermediate · 🟠 advanced · 🔴 expert

## Examples

The 20 programs in [`examples/`](examples/) compile and run as is (comments and
prompts are in French).

| File | Topic | File | Topic |
|---|---|---|---|
| `ex01_bonjour` | First question | `ex11_bas_niveau` | Tokens, logits, manual loop |
| `ex02_flux` | Streaming, speed | `ex12_rag` | Questions about your documents |
| `ex03_conversation` | Context, history, saving | `ex13_resume_long` | Map-reduce summarization |
| `ex04_chat_terminal` | Interactive assistant | `ex14_outils` | Agent with function calling |
| `ex05_parametres` | Sampling settings | `ex15_autograd` | Gradients, regression, XOR |
| `ex06_pieces_jointes` | CSV, code, images, PDF | `ex16_tokeniseur` | Training a tokenizer |
| `ex07_json` | Structured extraction | `ex17_creer_modele` | Building a model end to end |
| `ex08_image` | Image (SVG → BMP), drawing | `ex18_lora` | LoRA fine-tuning |
| `ex09_audio` | Speech synthesis, melody | `ex19_gguf_quantification` | Inspection, quantization |
| `ex10_fichier` | File generation | `ex20_reprise_entrainement` | Checkpoints, resuming |

## Correctness

The engine has been checked against the reference implementation,
[llama.cpp](https://github.com/ggml-org/llama.cpp):

- **Tokenizers**: llama.cpp's official test cases (Llama 3, Llama 2, Qwen 2,
  GPT-2, Phi-3, DeepSeek) produce exactly the same tokens.
- **Dequantization**: bit-identical to the `gguf-py` reference, for every format.
- **Inference**: greedy generation is identical to llama.cpp token for token
  (llama and qwen2 architectures; F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K formats).
- **Export**: models created, quantized or merged by nimllm load in llama.cpp and
  give the same answers there.
- **Automatic differentiation**: every operation is checked with finite differences.

```sh
nim c -r tests/t_complet.nim                            # end to end, self-contained
nim c -r tests/t_grad.nim                               # gradients
LLAMA_CPP=~/src/llama.cpp nim c -r tests/t_tok.nim      # tokenizers vs llama.cpp
```

## Architecture

```
src/
├── nimllm.nim            main module (imports everything)
└── nimllm/
    ├── gguf.nim          GGUF reading / writing
    ├── quant.nim         quantized formats, integer dot products
    ├── tokenizer.nim     BPE and SentencePiece
    ├── model.nim         inference: KV cache, RoPE, grouped-query attention, on-the-fly LoRA
    ├── sampler.nim       sampling
    ├── chat.nim          conversation, templates, output formats
    ├── attachments.nim   attachments (with inflate.nim for PNG and PDF)
    ├── image.nim         BMP/PPM images, drawing, SVG rendering
    ├── audio.nim         WAV, formant speech synthesis, melodies
    ├── autograd.nim      differentiable tensors
    ├── nn.nim            trainable Transformer, LoRA, export
    ├── train.nim         optimizers, datasets, training loop
    ├── tokentrain.nim    BPE tokenizer training
    └── parallel.nim      thread pool
```

## Limitations

- **CPU only**, no GPU. On 2 cores, a model the size of Llama 3.2 1B (Q4_K_M)
  generates about 7 tokens/s, versus about 13 for llama.cpp. Prompt processing is
  also slower than in llama.cpp.
- **Image and audio input**: Llama 3.2 1B/3B is a text-only model. Attached files
  are *described* to it (size, colors, ASCII preview, duration); it does not
  perceive them.
- **Image output**: the model writes SVG, which nimllm rasterizes. Drawing
  quality depends on the size of the model.
- **Audio output**: formant speech synthesis, intelligible but robotic. The
  pronunciation rules cover French and (more roughly) English.
- **Training**: realistic on a CPU for small models and for LoRA. On 2 cores, a
  LoRA step on Llama 3.2 1B (128-token batch) takes about 30 s.
- **Not supported**: Gemma, Phi-3, Mixture-of-Experts models, multimodal models,
  IQ* formats (i-quants).

## Contributing

Bug reports and suggestions are welcome in the issues. For a model-related
problem, please include the output of:

```nim
echo openGguf("model.gguf").describe()
```

Before submitting a change, make sure `nim c -r tests/t_complet.nim` and
`nim c -r tests/t_grad.nim` pass. All contributions are published under the
terms of [`LICENCE.md`](LICENCE.md).

## License

nimllm is distributed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**
(© sonaliwan.fr). A few third-party portions keep their own licenses (MIT,
Unicode v3, zlib): details are in [`LICENCE.md`](LICENCE.md) and the full texts
in [`LICENSES/`](LICENSES/).

**Commercial use**: a commercial license can be purchased from **sonaliwan.fr**
at [metalab@sonaliwan.fr](mailto:metalab@sonaliwan.fr).

Models used with nimllm have their own licenses, for example Meta's *Llama 3.2
Community License*.

## Acknowledgements

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp): GGUF format,
  quantization formats and tokenization algorithms (MIT).
- [GPT-2](https://github.com/openai/gpt-2): byte-level BPE encoding.
- [zlib / puff.c](https://github.com/madler/zlib): DEFLATE decoding.
- [Unicode Character Database](https://www.unicode.org/ucd/): character categories.
