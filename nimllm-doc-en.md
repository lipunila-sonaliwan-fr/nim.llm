# nimllm

**A language model library written 100% in Nim, with no external dependencies**:
no llama.cpp, no Python, no BLAS, no C library. All you need is the Nim compiler
and a model file in GGUF format (for example *Llama 3.2 1B Instruct*).

```nim
import nimllm

let model = loadModel("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let chat = newChat(model, system = "You are a concise assistant.")
echo chat.ask("What is the capital of Japan?").text
```

## What the library does

| Area | Features |
|---|---|
| **Models** | GGUF reading (memory-mapped), `llama` architectures (Llama 2/3/3.1/3.2, Mistral, TinyLlama…), `qwen2`, `qwen3`; weights F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K |
| **Tokenizers** | byte-level BPE (Llama 3, Qwen, GPT-2) and SentencePiece (Llama 2, Mistral), validated against llama.cpp's test suites |
| **Chat** | system prompt (context), multi-turn history, templates (Llama 3, ChatML, Mistral, Llama 2, Gemma, Phi-3), KV cache reuse, context overflow handling, save/resume, undo/regenerate/continue |
| **Generation** | token-by-token streaming, temperature, top-k, top-p, min-p, penalties, seed, stop strings, logit bias |
| **Attachments** | text, code, CSV, JSON, Markdown, HTML, PDF (text extraction), PNG/BMP/PPM (colors + preview), JPEG/GIF/WebP (dimensions), WAV |
| **Output formats** | text, Markdown, JSON (validated, with retries), **image** (SVG → rasterized BMP), **audio** (WAV via speech synthesis), **file** |
| **Tools** | embeddings, semantic search (RAG), perplexity, function calling / agents |
| **Training** | tensors with automatic differentiation, AdamW/SGD, cosine schedule, clipping, accumulation, checkpoints |
| **Fine-tuning** | LoRA on an existing quantized model (Llama 3.2…), applied on the fly or merged into a new GGUF |
| **Creation** | BPE tokenizer training, configurable Llama architecture, pre-training, chat fine-tuning, GGUF export (readable by llama.cpp, Ollama, LM Studio), quantization |

## Quick start

```sh
# 1. a GGUF model (example: Llama 3.2 1B Instruct quantized Q4_K_M, ~0.8 GB)
mkdir -p models   # put the .gguf file there (see chapter 01 — Installation)

# 2. an example
nim c -r examples/ex01_bonjour.nim models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# 3. a complete interactive assistant
nim c -r examples/ex04_chat_terminal.nim models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# 4. build your own model from scratch (no download needed)
nim c -r examples/ex17_creer_modele.nim
```

## Documentation

The documentation, in **`docs/`**, is progressive: each chapter builds on the
previous one, from a first "hello" to creating a model.

1. **Installation**
2. **First steps**
3. **Conversation and context**
4. **Generation settings**
5. **Attachments**
6. **Output formats: text, JSON, image, audio, file**
7. **Practical use cases: RAG, summarization, agents, extraction…**
8. **Under the hood: tokens, logits, cache, embeddings**
9. **Machine learning basics (autograd)**
10. **Fine-tuning an existing model with LoRA**
11. **Building a new model**
12. **GGUF and quantization**
13. **Performance and troubleshooting**
14. **API reference**

The 20 programs in [`examples/`](examples/) compile and run as is.

## Layout

```
nimllm/
├── src/nimllm.nim            main module (imports everything)
├── src/nimllm/
│   ├── gguf.nim              GGUF reading / writing
│   ├── quant.nim             quantized formats, integer dot products
│   ├── tokenizer.nim         BPE and SentencePiece
│   ├── model.nim             inference engine (KV cache, RoPE, GQA, on-the-fly LoRA)
│   ├── sampler.nim           sampling
│   ├── chat.nim              conversation, output formats
│   ├── attachments.nim       attachments; inflate.nim (zlib) for PNG/PDF
│   ├── image.nim             images, drawing, SVG rendering
│   ├── audio.nim             WAV, speech synthesis, melodies
│   ├── autograd.nim          differentiable tensors
│   ├── nn.nim                trainable Transformer, LoRA, export
│   ├── train.nim             optimizers, data, training loop
│   ├── tokentrain.nim        BPE tokenizer training
│   └── parallel.nim          thread pool
├── examples/                 20 commented programs
├── docs/                     progressive documentation
└── tests/                    tests (gradients, tokenizers, GGUF...)
```

## Correctness

The engine has been checked against the reference implementation, **llama.cpp**:

* tokenizers: 100% of llama.cpp's official test cases (Llama 3, Llama 2/SPM, Qwen 2,
  GPT-2, Phi-3, DeepSeek) produce exactly the same tokens;
* dequantization: bit-identical to the `gguf-py` reference for every format;
* inference: greedy generation identical to llama.cpp token for token (llama and
  qwen2 architectures, F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K formats);
* export: models created or merged by nimllm load in llama.cpp and give the same
  answers there;
* automatic differentiation: every operation is validated with finite differences.

## Limitations (honestly)

* **Speed**: CPU only (no GPU). On 12 cores, a model the size of Llama 3.2 1B Q4_K_M
  generates about 65 to 70 tokens/s (on a Mac Mini M5 Pro).
* **Vision and audio input**: Llama 3.2 1B/3B is a text-only model. Attached images
  and sounds are *described* to it (dimensions, colors, preview, duration); it does
  not perceive them.
* **Image output**: the model writes SVG, which nimllm rasterizes; quality depends
  on how well the model can draw in SVG (modest for a 1-billion-parameter model).
* **Audio output**: formant speech synthesis, robotic but self-contained.
* **Training**: feasible on a CPU for small models and for LoRA; pre-training a
  model the size of Llama 3.2 would take thousands of GPUs.
* Unsupported architectures: Gemma, Phi-3 (fused weights), Mixture-of-Experts,
  multimodal models; IQ* formats (i-quants).

## License

nimllm is distributed under **CC BY-NC-SA 4.0** (© sonaliwan.fr), except for a few
third-party portions under the MIT, Unicode v3 and zlib licenses, detailed in
`LICENCE.md`. A **commercial license** can be purchased from sonaliwan.fr
(metalab@sonaliwan.fr). The models you use have their own licenses (for example
Meta's *Llama 3.2 Community License*).


---

# nimllm documentation

This documentation is meant to be read in order: each chapter introduces a few new
concepts and builds on the previous ones. Every chapter contains **complete**
programs that you can copy into a `.nim` file and run as is.

## Learning path

| Level | Chapter | You will learn to… |
|---|---|---|
| Beginner | **01 — Installation** | install Nim, compile, get a GGUF model |
| Beginner | **02 — First steps** | ask a question, display the answer live |
| Beginner | **03 — Conversation and context** | define a role, chat, save, manage memory |
| Intermediate | **04 — Generation settings** | control creativity, length, repetition, reproducibility |
| Intermediate | **05 — Attachments** | attach text, CSV, code, PDF, images, sounds |
| Intermediate | **06 — Output formats** | get JSON, an image, an audio file, a file |
| Intermediate | **07 — Practical use cases** | RAG, long-document summarization, tool-using agents, classification |
| Advanced | **08 — Under the hood** | tokens, logits, KV cache, templates, embeddings, perplexity |
| Advanced | **09 — Machine learning basics** | tensors, gradients, optimizers, neural networks |
| Expert | **10 — Fine-tuning with LoRA** | specialize Llama 3.2 on your data |
| Expert | **11 — Building a new model** | tokenizer, architecture, pre-training, chat, export |
| Expert | **12 — GGUF and quantization** | inspect, write and quantize model files |
| All | **13 — Performance and troubleshooting** | go faster, understand errors |
| All | **14 — API reference** | every public function |

## Conventions

* Programs start with a `# file: name.nim` comment.
* The model path is read from the command line or from the `NIMLLM_MODELE`
  environment variable (this is the variable's actual name):

  ```sh
  export NIMLLM_MODELE=$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  nim c -r my_program.nim
  ```

* **Always** compile with `-d:release` (or use the provided `config.nims`): in
  debug mode, computation is 10 to 30 times slower.
* The library's own messages (error messages, training logs, `describe()`
  summaries, attachment descriptions) are in French. Sample outputs reproduced in
  this documentation are therefore shown in French, with an explanation.
* For English conversations, create chats with `lang = "en"`: the instructions
  nimllm adds for JSON, image, audio and file outputs, as well as the synthetic
  voice, will then be in English (the default is `"fr"`).

## Minimal vocabulary

| Term | Meaning |
|---|---|
| **LLM** | *Large Language Model*: a neural network that predicts how a text continues |
| **token** | a piece of text (word, part of a word, punctuation) handled by the model |
| **context** | all the tokens the model "sees" (prompt + history + answer) |
| **system prompt** | a standing instruction that defines the model's role and behavior |
| **GGUF** | model file format (weights + tokenizer + metadata) |
| **quantization** | weight compression (e.g. 4 bits instead of 32) |
| **logits** | raw scores the model assigns to each possible token |
| **sampling** | the way the next token is chosen from the logits |
| **LoRA** | lightweight fine-tuning technique: only small added matrices are trained |


---

# 01 — Installation

## 1.1 Installing Nim

nimllm requires **Nim 2.0 or later** and a C compiler (gcc or clang).

```sh
# Linux / macOS: official installer
curl https://nim-lang.org/choosenim/init.sh -sSf | sh
# or with your package manager: apt install nim / brew install nim
nim -v        # should print 2.x
```

On Windows, use the installer from <https://nim-lang.org/install.html> (it ships MinGW).

## 1.2 Installing nimllm

nimllm does not depend on any package. Two options:

```sh
# a) install as a Nimble package (from the project folder)
cd nimllm
nimble install

# b) without installing: give the source path at compile time
nim c -d:release --path:path/to/nimllm/src my_program.nim
```

The `examples/` folder contains a `config.nims` that sets everything up for you
(source path, `-d:release`, SIMD instructions):

```sh
nim c -r examples/ex01_bonjour.nim
```

For your own projects, create a `config.nims` next to your sources:

```nim
# file: config.nims
switch("path", "/path/to/nimllm/src")       # not needed if installed with nimble
switch("define", "release")                  # essential for speed
switch("passC", "-march=native")             # uses AVX2/AVX-512 when available
```

> **Important.** Without `-d:release`, Nim compiles in debug mode with every check
> enabled: inference is then 10 to 30 times slower.

## 1.3 Getting a model

nimllm reads **GGUF** files, llama.cpp's widely used format. To get started, the
recommended model is **Llama 3.2 1B Instruct in Q4_K_M** (~0.8 GB, runs with 2 GB
of memory). The 3B (~2 GB) answers much better, at the cost of speed divided by
about 2.5.

Possible sources:

* **Hugging Face**: search for "Llama-3.2-1B-Instruct-GGUF" and take the
  `...Q4_K_M.gguf` file. Meta's official repositories require you to accept the
  *Llama 3.2 Community License*.
* **Ollama**: if you have already run `ollama pull llama3.2:1b`, the model is a
  GGUF file in `~/.ollama/models/blobs/` (the largest `sha256-…` file). You can
  use it directly with nimllm.
* **Your own model**: chapter 11 shows how to build one without downloading
  anything.

Other compatible models: Llama 3.1 8B, Mistral 7B, Qwen 2.5 (0.5B to 7B), Qwen 3,
TinyLlama, SmolLM, etc., in GGUF F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q4_0.

Store the model, for example, in `models/`:

```
my_project/
├── config.nims
├── models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
└── hello.nim
```

## 1.4 Checking the installation

```nim
# file: check.nim
## Prints the characteristics of a GGUF model.
import std/os
import nimllm

let path = if paramCount() >= 1: paramStr(1)
           else: getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let model = loadModel(path, verbose = true)
echo model.describe
echo "Compute threads: ", numThreads()
echo "Chat template: ", detectTemplate(model.tokenizer)
```

```sh
nim c -r check.nim models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Typical output for Llama 3.2 1B (exact values depend on the file). The first four
lines come from `describe`, whose labels are in French: *couches* = layers,
*têtes* = heads, *têtesKV* = KV heads, *dimTête* = head dimension, *contexte* =
context, *paramètres* = parameters, *poids* = weights, *sortie* = output.

```
Modèle Llama 3.2 1B Instruct [llama]
  couches=16 dim=2048 ffn=8192 têtes=32 têtesKV=8 dimTête=64
  vocab=128256 contexte=131072 rope_base=500000.0 paramètres=1235.8 M
  poids : gtQ4_K (attention), gtQ6_K (sortie)
Compute threads: 8
Chat template: llama3
```

## 1.5 Memory requirements

| Item | Llama 3.2 1B Q4_K_M | Llama 3.2 3B Q4_K_M |
|---|---|---|
| Weights (mapped, shared) | ~0.8 GB | ~2.0 GB |
| KV cache per 4096-token context | ~0.25 GB | ~0.9 GB |
| LoRA fine-tuning (128-token batch) | +1.3 GB | +3 GB |

Weights are *mapped* from the file: they are only loaded into memory on demand,
and several conversations share them.

**Next:** **02 — First steps**


---

# 02 — First steps

Goal: ask a model a question and use the answer.

## 2.1 Three objects to know

```
LlmModel  ──►  Chat (conversation)  ──►  Reply (answer)
 weights        history, settings          text, files, statistics
 tokenizer      KV cache (LlmContext)
```

* `loadModel(path)` opens the GGUF file and returns an **`LlmModel`**.
* `newChat(model)` creates a **conversation** (`Chat`) that keeps the history.
* `chat.ask(question)` returns a **`Reply`**; the text is in `.text`.

## 2.2 The minimal program

```nim
# file: hello.nim
import std/os
import nimllm

let path = if paramCount() >= 1: paramStr(1)
           else: getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

let model = loadModel(path)                       # 1. load the model
let chat = newChat(model, lang = "en")            # 2. open a conversation
let r = chat.ask("What is the capital of France?")   # 3. ask the question
echo r.text                                       # 4. use the answer
```

```sh
nim c -r -d:release hello.nim models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
# The capital of France is Paris.
```

## 2.3 Displaying the answer as it is generated

For long answers, display each piece as soon as it is produced with the `onToken`
parameter. The function receives a piece of text (complete UTF-8) and returns
`true` to continue.

```nim
# file: stream.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, maxTokens = 400, lang = "en")

let r = chat.ask("Explain photosynthesis to an 8-year-old.",
  onToken = proc (piece: string): bool =
    stdout.write piece
    stdout.flushFile()
    true)
echo ""
echo "(", r.completionTokens, " tokens in ", r.seconds.int, " s, ",
     r.tokensPerSecond.int, " tokens/s, stop: ", r.stopReason, ")"
```

## 2.4 What a reply contains

| Field | Content |
|---|---|
| `text` | final text, cleaned up according to the requested format |
| `raw` | raw text as produced by the model |
| `files` | files created (images, audio, files) |
| `json` | parsed JSON object (JSON format) |
| `promptTokens` | size of the prompt sent (system + history + question) |
| `completionTokens` | number of tokens generated |
| `stopReason` | `srEndOfText` (natural end), `srMaxTokens`, `srStopString`, `srContextFull`, `srCallback` |
| `seconds`, `tokensPerSecond` | duration and speed |

If `stopReason == srMaxTokens`, the answer was cut off: increase `maxTokens` or
call `chat.continueReply()` to extend it.

```nim
# file: extend.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, maxTokens = 40, lang = "en")   # deliberately short
var r = chat.ask("Describe the four seasons.")
while r.stopReason == srMaxTokens:
  echo "... (cut off, continuing)"
  r = chat.continueReply(maxTokens = 60)
echo r.text                                       # full text
```

## 2.5 Handling errors

Functions raise explicit exceptions:

| Exception | Typical cause |
|---|---|
| `GgufError` | file missing, truncated, or not a GGUF file |
| `ModelError` | unsupported architecture or weight format |
| `ChatError` | message longer than the context, no JSON/SVG found in the answer |
| `IOError` | attachment not found |

```nim
# file: errors.nim
import nimllm

try:
  let m = loadModel("missing.gguf")
  discard m
except GgufError as e:
  echo "Cannot load the model: ", e.msg
```

## 2.6 Exercises

1. Change `hello.nim` to read the question from the command line (`paramStr(2)`).
2. In `stream.nim`, stop generation as soon as the answer contains the word
   "chlorophyll" (return `false`).
3. Print `r.promptTokens`: why is it much larger than the number of words in the
   question? (hint: chapter 8, chat templates).

**Next:** **03 — Conversation and context**


---

# 03 — Conversation and context

Goal: define the **context** (role, instructions), hold a multi-turn conversation,
and control the conversation's memory.

## 3.1 The system prompt

The system prompt is a standing instruction placed before the history. It is the
main way to define the model's behavior: role, tone, language, format, rules,
reference knowledge.

```nim
# file: role.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

let guide = newChat(model, lang = "en", system = """
You are Léa, a tour guide in Lyon.
- You answer in English, in 3 sentences maximum.
- You always suggest a specific address.
- If someone mentions another city, you politely bring the conversation back to Lyon.""")

echo guide.ask("Where can I eat a local specialty?").text
echo guide.ask("And in Marseille?").text
```

Tips for a good system prompt:

* be **explicit** ("3 sentences maximum" rather than "be brief");
* give **examples** of the expected format;
* put reference information (prices, opening hours…) in the system prompt or in
  an attachment;
* for a small model (1B), prefer short, simple instructions.

The system prompt can be changed at any time: `chat.system = "..."`.

## 3.2 Multi-turn conversation

Each call to `ask` adds the question and the answer to `chat.history`. The model
therefore sees the whole conversation:

```nim
# file: dialogue.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, lang = "en", system = "You are a patient math teacher.")

for question in ["What is a prime number?",
                 "Give me the first five.",
                 "And the next one after the last one you listed?"]:
  echo "Student: ", question
  echo "Teacher: ", chat.ask(question).text, "\n"

# Inspecting the history
for m in chat.history:
  echo "[", m.role, "] ", m.content[0 ..< min(60, m.content.len)]
echo chat.stats
```

### Efficiency: the KV cache

The model remembers the computations already done on the beginning of the
conversation (the *key/value cache*). At each turn, nimllm compares the new prompt
with the cache contents and only computes the new part: long conversations stay
responsive.

## 3.3 Managing the history

| Operation | Effect |
|---|---|
| `chat.reset()` | clears the history (keeps the system prompt) |
| `chat.undo()` | removes the last question/answer exchange |
| `chat.regenerate()` | replaces the last answer with a new draw |
| `chat.continueReply()` | extends the last answer |
| `chat.add(role, text)` | adds a message without generating anything |
| `chat.history` | the list of messages (can be edited directly) |

```nim
# file: history.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, lang = "en")

discard chat.ask("Suggest a name for a cat.")
echo "1st draw: ", chat.history[^1].content
echo "2nd draw: ", chat.regenerate().text      # another suggestion
chat.undo()                                     # forget this exchange
echo "Messages: ", chat.history.len             # 0

# Injecting a made-up history: useful to resume a session or to
# impose a style by example ("few-shot").
chat.add(roleUser, "Translate into French: cat")
chat.add(roleAssistant, "chat")
chat.add(roleUser, "Translate into French: dog")
chat.add(roleAssistant, "chien")
echo chat.ask("Translate into French: bird").text   # oiseau
```

## 3.4 Saving and resuming a conversation

```nim
# file: session.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

if fileExists("session.json"):
  let chat = newChat(model, lang = "en")
  chat.loadHistory("session.json")       # system + messages
  echo "Resuming (", chat.history.len, " messages)"
  echo chat.ask("What were we talking about?").text
  chat.saveHistory("session.json")
else:
  let chat = newChat(model, lang = "en", system = "You are a running coach.")
  echo chat.ask("I want to run a 10K in 2 months. Where do I start?").text
  chat.saveHistory("session.json")
  echo "Session saved; run the program again."
```

The JSON file is readable and editable:

```json
{
  "system": "You are a running coach.",
  "template": "llama3",
  "messages": [
    {"role": "user", "content": "I want to run a 10K..."},
    {"role": "assistant", "content": "Start with..."}
  ]
}
```

## 3.5 The context window

A model only sees a limited number of tokens: the **context window**.
`newChat(model, nCtx = 4096)` reserves a cache for 4096 tokens (Llama 3.2 accepts
up to 131,072, but every token costs memory: ~64 KiB for Llama 3.2 1B, ~224 KiB
for the 3B).

What happens when the conversation gets too long?

1. Before each answer, nimllm checks that *prompt + maxTokens* fits in `nCtx`;
2. if not, it removes the **oldest** exchanges (never the system prompt);
3. if the last question alone is too long, `ChatError` is raised.

```nim
# file: long_memory.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, nCtx = 512, maxTokens = 100, lang = "en")   # small window on purpose
chat.system = "Remember everything I tell you."
for i in 1 .. 15:
  discard chat.ask("Note number " & $i & ": I like the number " & $(i * 7) & ".")
  echo "turn ", i, ": ", chat.history.len, " messages kept, ", chat.stats
```

To keep a long-term memory anyway, there are two techniques:

* periodically **summarize** the history and put it in the system prompt;
* store the information and **retrieve** it when needed (RAG, chapter 7).

```nim
# file: summarized_memory.nim
## When the history exceeds 10 messages, it is replaced by a summary.
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let baseSystem = "You are a personal assistant."
let chat = newChat(model, lang = "en", system = baseSystem)

proc compact(c: Chat) =
  if c.history.len < 10: return
  var transcript = ""
  for m in c.history: transcript.add $m.role & ": " & m.content & "\n"
  let summarizer = newChat(c.model, sampling = greedySampling(), maxTokens = 200, lang = "en")
  let summary = summarizer.ask("Summarize the important facts of this conversation " &
                               "as a short list:\n" & transcript).text
  c.system = baseSystem & "\n\nWhat you already know about the user:\n" & summary
  c.reset()

for msg in ["My name is Paul.", "I live in Rennes.", "I have two cats.",
            "I work as a nurse.", "My favorite dish is the galette.",
            "What is my name and where do I live?"]:
  echo "> ", msg
  echo chat.ask(msg).text
  compact(chat)
```

## 3.6 Several conversations in parallel

Weights are shared: each `Chat` only adds its own cache. You can therefore run
several independent conversations with a single loaded model.

```nim
# file: two_characters.nim
## Two characters talk to each other.
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let pirate = newChat(model, lang = "en", system = "You are a gruff pirate. One sentence per line.", maxTokens = 60)
let robot = newChat(model, lang = "en", system = "You are a very polite robot. One sentence per line.", maxTokens = 60)

var line = "Hello, who are you?"
for turn in 1 .. 3:
  line = pirate.ask(line).text
  echo "Pirate: ", line
  line = robot.ask(line).text
  echo "Robot:  ", line
```

## 3.7 Exercises

1. Write a spell checker: system prompt "Correct the text without comment" and
   `greedySampling()`.
2. Add a command to `session.nim` that deletes the session.
3. Measure the speed of the 2nd turn of a conversation (`r.seconds`) with and
   without `chat.reset()` between turns: observe the effect of the cache.

**Next:** **04 — Generation settings**


---

# 04 — Generation settings

Goal: understand how the model chooses its words, and tune creativity, precision,
length and reproducibility.

## 4.1 How a token is chosen

At each step, the model assigns a score (*logit*) to each of the ~128,000 tokens
in its vocabulary. These scores are turned into probabilities, then a token is
drawn through a chain of filters:

```
logits ─► repetition penalties ─► bias ─► temperature ─► top-k ─► top-p ─► min-p ─► draw
```

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | 0.7 | 0 = always the most likely; < 1 safer; > 1 more inventive |
| `topK` | 40 | keeps only the K best candidates (0 = disabled) |
| `topP` | 0.95 | keeps the smallest group totaling probability P (1 = disabled) |
| `minP` | 0.05 | removes candidates < minP × probability of the best one |
| `repeatPenalty` | 1.1 | > 1 discourages reusing recent tokens |
| `repeatLastN` | 64 | window of tokens watched by the penalty |
| `presencePenalty` | 0 | fixed penalty for any token that has already appeared |
| `frequencyPenalty` | 0 | penalty proportional to the number of occurrences |
| `seed` | -1 | random seed; -1 = random, ≥ 0 = reproducible |
| `logitBias` | empty | adds a bias to specific tokens (-100 = forbidden) |

Two presets: `defaultSampling()` (conversation) and `greedySampling()`
(deterministic: extraction, classification, code, calculation).

## 4.2 Choosing settings

```nim
# file: temperatures.nim
## Compares the same question at different temperatures.
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let question = "Suggest an original title for a detective novel."

for t in [0.0, 0.5, 0.9, 1.4]:
  var p = defaultSampling()
  p.temperature = t
  let chat = newChat(model, sampling = p, maxTokens = 30, lang = "en")
  echo "T=", t, ": ", chat.ask(question).text
```

Guidelines:

| Use | Recommended setting |
|---|---|
| Extraction, classification, JSON, calculation | `greedySampling()` or temperature 0–0.2 |
| Factual questions and answers | temperature 0.3–0.5 |
| Conversation | `defaultSampling()` (0.7) |
| Creative writing, brainstorming | temperature 0.9–1.1, topP 0.95 |
| Above 1.3 | text often incoherent |

## 4.3 Reproducibility

With a fixed seed, the same inputs always give the same output (on the same
machine with the same number of threads):

```nim
# file: seed.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
var p = defaultSampling()
p.seed = 1234
for attempt in 1 .. 2:
  let chat = newChat(model, sampling = p, maxTokens = 40, lang = "en")
  echo "Attempt ", attempt, ": ", chat.ask("Make up a proverb.").text   # identical
```

## 4.4 Length and stop strings

* `maxTokens` limits the length (1 token ≈ 0.75 word in English, a bit less in
  French);
* `stop` ends generation as soon as a string appears (it is removed from the
  result).

```nim
# file: stop.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, lang = "en")
chat.options.maxTokens = 200
chat.options.stop = @["4."]          # stops before the 4th item of a list
let r = chat.ask("List ten fruits, numbered.")
echo r.text
echo "Reason: ", r.stopReason        # srStopString

# One-off setting, for a single call:
echo chat.ask("One word for hello in Italian?", maxTokens = 5).text
```

## 4.5 Avoiding repetition

Small models tend to loop. Levers, from gentlest to strongest:

```nim
# file: repetition.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
var p = defaultSampling()
p.repeatPenalty = 1.2        # 1.05 to 1.3
p.repeatLastN = 128          # watches further back
p.frequencyPenalty = 0.3     # penalizes very frequent words
p.presencePenalty = 0.2      # encourages new topics
let chat = newChat(model, sampling = p, maxTokens = 300, lang = "en")
echo chat.ask("Write a short text about the sea.").text
```

Beware: too strong a penalty degrades grammar (the model avoids necessary words
such as "the" or "of").

## 4.6 Forbidding or favoring words

`logitBias` acts on **tokens**: first encode the word (with and without a leading
space, since "Paris" and "␣Paris" are different tokens).

```nim
# file: bias.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = model.tokenizer

var p = greedySampling()
for variant in ["yes", " yes", "Yes", " Yes"]:
  let ids = tok.encode(variant)
  if ids.len == 1: p.logitBias[ids[0]] = -100.0     # forbidden
let chat = newChat(model, sampling = p, maxTokens = 20, lang = "en")
echo chat.ask("Is the sky blue? Answer with one word.").text

# Favoring: a positive bias (+2 to +5) makes a token more likely.
```

## 4.7 Changing settings during a conversation

```nim
# file: switch.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, lang = "en")
echo chat.ask("Imagine a fantastic creature.").text   # creative
chat.setSampling(greedySampling())                     # precise from now on
echo chat.ask("Summarize your description in 5 words.").text
```

## 4.8 Exercises

1. Generate 5 slogans with the same question and seeds 1 to 5.
2. Find the temperature above which Llama 3.2 1B makes grammar mistakes on
   "Tell me about your day".
3. With `logitBias`, prevent the model from using words starting with the letter
   "e" (hard exercise: you need to go through the whole vocabulary with
   `tok.tokenToPiece(id)`).

**Next:** **05 — Attachments**


---

# 05 — Attachments

Goal: send documents along with a question (text, tables, code, PDF, images,
sounds).

## 5.1 How it works

Llama 3.2 (1B/3B) is a **text-only** model. nimllm therefore converts each
attachment into text, inserted in the message after the question:

| Type | What the model receives |
|---|---|
| Text, Markdown, code, CSV, JSON, HTML, XML, YAML… | the full content, in a code block |
| PDF | the text extracted from the pages ("text" PDFs; not scans) |
| PNG, BMP, PPM | dimensions, brightness, dominant colors, ASCII preview |
| JPEG, GIF, WebP | format and dimensions |
| WAV | sample rate, channels, duration |
| Other binary | size and first bytes |

Beyond 12,000 characters, the content is truncated (`maxChars` parameter of
`toPrompt`). Also keep the context window in mind: a 30,000-character file is
~8,000 tokens; create the conversation with a large enough `nCtx` (e.g. 16384).

## 5.2 Attaching files

```nim
# file: attach_files.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, nCtx = 8192, maxTokens = 400, lang = "en")

writeFile("budget.csv", """item,planned,actual
rent,900,900
groceries,400,465
transport,120,98
leisure,150,210
""")

let r = chat.ask("Which items are over the planned budget, and by how much?",
                 attachments = @[attach("budget.csv")])
echo r.text
```

`attach(path)` detects the type from the content (signature) and the extension.

## 5.3 Attachments created in memory

```nim
# file: in_memory.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, maxTokens = 300, lang = "en")

# text built by the program
let report = attachText("report.md", "# Incident 42\nServer down from 2:00 pm to 2:20 pm.\nCause: disk full.")
# raw bytes (e.g. received over the network): the type is detected
let data = attachData("measures.json", """{"temperature": [18.5, 19.2, 21.0], "unit": "°C"}""")

echo chat.ask("Write an apology message to customers based on the report, " &
              "and give the average temperature.", attachments = @[report, data]).text
```

## 5.4 Seeing what the model receives

`toPrompt` shows the exact text that is inserted. It is the number one
diagnostic tool:

```nim
# file: preview.nim
import nimllm

let img = renderSvg("""<svg viewBox="0 0 60 40"><rect width="60" height="40" fill="navy"/>
  <circle cx="30" cy="20" r="12" fill="yellow"/></svg>""", 120, 80)
img.writeBmp("flag.bmp")
let att = attach("flag.bmp")
echo att.kind, " ", att.width, "x", att.height
echo att.toPrompt()
```

The description is generated by the library, in French (*Pièce jointe* =
attachment, *Luminosité moyenne* = average brightness, *Couleurs dominantes* =
dominant colors, *bleu foncé* = dark blue, *jaune* = yellow, *Aperçu* = preview,
*clair* = light, *sombre* = dark):

```
akImage 120x80
### Pièce jointe : flag.bmp (image/bmp)
Image BMP de 120×80 pixels.
Luminosité moyenne : 21 %
Couleurs dominantes : bleu foncé (80 %), jaune (18 %)
Aperçu (48×16, clair = espace, sombre = @) :
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@%...........@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@%...............@@@@@@@@@@@@@@@@
...
```

## 5.5 PDF documents

```nim
# file: pdf.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let path = if paramCount() >= 1: paramStr(1) else: "invoice.pdf"
let pdf = attach(path)
echo "Extracted text: ", pdf.text.len, " characters"
let chat = newChat(model, nCtx = 8192, lang = "en")
echo chat.ask("What is the total amount and the due date?", attachments = @[pdf]).text
```

Extraction handles compressed streams (FlateDecode) and the usual text operators.
Limitations: scanned PDFs (images), fonts with custom encodings (some PDFs made by
desktop-publishing software), multi-column layouts. When in doubt, print
`pdf.text`.

## 5.6 Images: what can be extracted

The model does not see the image, but the description provided is enough for
simple questions (colors, brightness, rough shapes, orientation). For a finer
analysis, you can compute information yourself and attach it as text:

```nim
# file: image_analysis.nim
## Computes statistics on an image and has them commented on.
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let path = if paramCount() >= 1: paramStr(1) else: "photo.png"
let att = attach(path)
if att.image == nil:
  quit "format not decoded (PNG, BMP or PPM expected): " & path

# Share of "sky" (blue) pixels in the upper half
let img = att.image
var blues = 0
for y in 0 ..< img.height div 2:
  for x in 0 ..< img.width:
    let (r, g, b) = img.getPixel(x, y)
    if b.int > r.int + 30 and b.int > g.int: inc blues
let ratio = blues * 100 div max(1, img.width * img.height div 2)

let chat = newChat(model, lang = "en")
let info = attachText("measures.txt", "Blue pixels in the upper half: " & $ratio & " %")
echo chat.ask("Was this photo taken outdoors in good weather? Explain.",
              attachments = @[att, info]).text
```

PNG files (8/16 bits, grayscale, color, palette, transparency, non-interlaced) are
decoded entirely in Nim (`decodePng`), as are 24/32-bit BMP and PPM files.

## 5.7 Several documents, comparison

```nim
# file: compare.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let v1 = attachText("contract_v1.txt", "Term: 12 months. Notice: 1 month. Price: €30/month.")
let v2 = attachText("contract_v2.txt", "Term: 24 months. Notice: 3 months. Price: €27/month.")
let chat = newChat(model, sampling = greedySampling(), lang = "en")
echo chat.ask("List the differences between the two versions of the contract.",
              attachments = @[v1, v2], format = ofMarkdown).text
```

## 5.8 Good practices

* Ask the question **before** the attachments (that is what `ask` does) and be
  precise about what to extract from them.
* For a large document, split it (chapter 7: map-reduce summarization and RAG).
* Attachments stay in the history: call `chat.reset()` between two unrelated
  documents to free the context.
* Attachments are never interpreted as control tags: a document containing
  `<|eot_id|>` cannot hijack the conversation.

**Next:** **06 — Output formats**


---

# 06 — Output formats: text, JSON, image, audio, file

Goal: receive the answer in the form you want.

## 6.1 Overview

The `format` parameter of `ask` chooses the form of the answer:

| Format | What nimllm does | Result |
|---|---|---|
| `ofText` (default) | nothing special | `r.text` |
| `ofMarkdown` | asks for Markdown formatting | `r.text` |
| `ofJson` | requires JSON, extracts it, validates it, retries if needed | `r.json` (+ `r.text`) |
| `ofImage` | has the model write SVG, extracts it and rasterizes it | `r.files` = `.svg` + `.bmp` |
| `ofAudio` | asks for simple sentences, then synthesizes them | `r.files` = `.wav` |
| `ofFile` | asks for the raw content of a file and writes it | `r.files` = file |

For image, audio and file, `outPath` gives the file name (otherwise a timestamped
name is created in the current folder).

The instructions nimllm adds to the question for these formats follow the chat's
language: create the chat with `lang = "en"` to get them in English.

## 6.2 Markdown

```nim
# file: markdown.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, maxTokens = 500, lang = "en")
let r = chat.ask("Compare the train and the plane for a London–Edinburgh trip.", format = ofMarkdown)
writeFile("comparison.md", r.text)
echo r.text
```

## 6.3 JSON: answers your program can use

`schema` describes the expected structure (free text, read by the model). nimllm:

1. adds the instruction "answer only in JSON" and the schema;
2. extracts the JSON from the answer (even when surrounded by text or ``` fences);
3. if that fails, retries `chat.jsonRetries` times (default 2) with a lower
   temperature, then raises `ChatError`.

```nim
# file: json_extraction.nim
import std/[os, json]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, sampling = greedySampling(), maxTokens = 300, lang = "en")

let listing = """For sale: Moustache electric bike, 2021, 3,200 km, 500 Wh battery,
very good condition, €1,450 or best offer, can be seen in Angers. Tel 06 00 00 00 00."""

let r = chat.ask("Extract the information from this listing:\n" & listing,
  format = ofJson,
  schema = """{"item": str, "brand": str, "year": int, "mileage_km": int,
               "price_eur": int, "city": str, "negotiable": bool}""")

let j = r.json
echo j.pretty
echo "Price: €", j{"price_eur"}.getInt, " (negotiable: ", j{"negotiable"}.getBool, ")"

# Direct conversion into a Nim object
type Listing = object
  item, brand, city: string
  year, mileage_km, price_eur: int
  negotiable: bool
try:
  let a = j.to(Listing)
  echo a.brand, " from ", a.year, " in ", a.city
except CatchableError:
  echo "A field is missing or has an unexpected type."
```

Shortcut: `chat.askJson(question, schema)` returns the `JsonNode` directly.

Tips:

* use `greedySampling()` or a temperature ≤ 0.3;
* give a **concrete** schema (field names + types or example values);
* always validate fields (`j{"key"}.getStr("default")` does not crash if the key
  is missing);
* `extractJson(text)` can be used on its own on any text.

## 6.4 Image

A language model does not produce pixels. nimllm therefore has it write a
**vector SVG** image (text), then rasterizes it to BMP itself. You get two files:
the `.svg` (sharp at any size, opens in a browser) and the `.bmp` (universal
bitmap image).

```nim
# file: draw.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, maxTokens = 1500, lang = "en")
chat.imageSize = 512                       # width of the BMP produced

let r = chat.ask("Draw a sailboat on the sea at sunset.",
                 format = ofImage, outPath = "boat.svg")
echo "Files: ", r.files                    # @["boat.svg", "boat.bmp"]
echo "SVG code: ", r.text.len, " characters"
```

The renderer supports: `rect` (rounded corners), `circle`, `ellipse`, `line`,
`polyline`, `polygon`, `path` (M, L, H, V, C, S, Q, T, A, Z, absolute and
relative), `g` groups, `transform` (translate, scale, rotate, matrix, skew), named
colors / `#rgb` / `#rrggbb` / `rgb()`, `fill`, `stroke`, `stroke-width`,
opacities, `fill-rule`, `style` attribute. Text (`<text>`) and gradients are not
drawn (a gradient becomes a neutral gray).

Quality: a 1B model draws simple shapes; a 3B or 8B model does much better.
Improve the result by describing the composition ("a yellow circle at the top
right, a blue rectangle at the bottom…").

### Rendering SVG or drawing without an LLM

```nim
# file: render.nim
import std/os
import nimllm

# Rendering an existing SVG at the desired size
if not fileExists("boat.svg"):
  writeFile("boat.svg", """<svg viewBox="0 0 100 60"><rect width="100" height="60" fill="lightblue"/>
    <polygon points="20,45 80,45 70,55 30,55" fill="brown"/><polygon points="50,5 50,43 25,43" fill="white"/></svg>""")
let svg = readFile("boat.svg")
renderSvg(svg, width = 1024).writeBmp("boat_large.bmp")

# Programmatic drawing (anti-aliasing through supersampling)
let c = newCanvas(400, 300, rgb(240, 248, 255))
c.fillRect(0, 220, 400, 80, rgb(30, 110, 200))                  # sea
c.fillCircle(320, 70, 40, rgb(255, 170, 0))                      # sun
c.fillPolygon(@[(150.0, 200.0), (200.0, 80.0), (200.0, 200.0)], rgb(250, 250, 250))  # sail
c.strokePolyline(@[(120.0, 205.0), (230.0, 205.0), (210.0, 225.0), (140.0, 225.0)],
                 3, rgb(90, 50, 20), closed = true)              # hull
c.finish().writeBmp("drawing.bmp")
```

## 6.5 Audio

The answer is generated as text (with an instruction to write simple sentences
without formatting), then read by the built-in speech synthesizer and written as
WAV (16 kHz, mono, 16 bits).

```nim
# file: speak.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, maxTokens = 120)
chat.lang = "en"             # "fr" or "en": language of the instructions and of the voice
chat.voicePitch = 120        # voice pitch in Hz

let r = chat.ask("Wish me a nice day in two sentences.",
                 format = ofAudio, outPath = "nice_day.wav")
echo r.text
echo r.files[0], ": ", readWavInfo(r.files[0]).durationSec, " s"
```

The synthesizer works with **formants** (a simulation of the resonances of the
vocal tract) with simplified French and English pronunciation rules: the voice is
robotic but needs no extra model. The French rules are the most complete; English
pronunciation is rougher. Numbers are read out in words (`numberToEnglish(1971)`
→ "one thousand nine hundred seventy one", `numberToFrench(1971)` → "mille neuf
cent soixante et onze").

Audio functions you can use directly:

```nim
# file: sounds.nim
import nimllm

speak("Attention, the train is about to leave.", lang = "en", pitch = 100, speed = 0.9).writeWav("announcement.wav")

var message = speak("First point.", lang = "en")
message.silence(0.4)                                  # 0.4 s pause
message.concat(speak("Second point.", lang = "en"))
message.writeWav("two_points.wav")

renderMelody("E4 D4 C4 D4 E4 E4 E4:2 D4 D4 D4:2 E4 G4 G4:2", bpm = 140).writeWav("tune.wav")
echo noteFrequency("A4")                              # 440.0

let sound = readWav("announcement.wav")               # reading (16-bit PCM)
echo sound.duration, " s at ", sound.sampleRate, " Hz"
```

## 6.6 File

For any other text format (CSV, code, configuration, HTML…), `ofFile` asks for
the raw content, removes any ``` fences and writes the file:

```nim
# file: generate_files.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
var p = defaultSampling()
p.temperature = 0.2
let chat = newChat(model, sampling = p, maxTokens = 1000, lang = "en")

let r = chat.ask("A simple HTML page presenting a bakery, with a title and a list of 3 products.",
                 format = ofFile, outPath = "bakery.html")
echo "Created: ", r.files[0]

chat.reset()
discard chat.ask("A shell script that backs up the ~/Documents folder into a dated archive.",
                 format = ofFile, outPath = "backup.sh")
```

## 6.7 Combining attachments and formats

Attachment as input + format as output = document transformation:

```nim
# file: transform.nim
import std/[os, json]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, sampling = greedySampling(), maxTokens = 800, lang = "en")

writeFile("contacts.txt", "Alice Martin, alice@example.com, Lyon\nBruno Petit, bruno@example.com, Lille\n")

let r = chat.ask("Convert this list to JSON.", attachments = @[attach("contacts.txt")],
                 format = ofJson, schema = """{"contacts": [{"name": str, "email": str, "city": str}]}""")
let contacts = r.json{"contacts"}          # {}: nil if the key is missing (no exception)
if contacts != nil and contacts.kind == JArray:
  for c in contacts: echo c{"name"}.getStr, " <", c{"email"}.getStr, ">"

chat.reset()
discard chat.ask("Make a bar chart of the number of contacts per city.",
                 attachments = @[attach("contacts.txt")], format = ofImage, outPath = "cities.svg")
```

**Next:** **07 — Practical use cases**


---

# 07 — Practical use cases

This chapter combines the previous concepts into complete programs for the most
common uses.

| Case | Techniques |
|---|---|
| 7.1 Classifying hundreds of texts | greedy, JSON, loop, cache |
| 7.2 Translating a file | splitting, system prompt |
| 7.3 Questions about your documents (RAG) | embeddings, search, attachments |
| 7.4 Summarizing a very long document | map-reduce, token counting |
| 7.5 Agent with tools | JSON, decision loop |
| 7.6 Generating training data | controlled creativity, JSONL |
| 7.7 A local HTTP service | std/asynchttpserver |

## 7.1 Classifying hundreds of texts

```nim
# file: classify_reviews.nim
## Classifies customer reviews (sentiment + topic) and produces a CSV.
import std/[os, json, strutils]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let reviews = @[
  "Fast delivery, product as described, I recommend it.",
  "The parcel arrived damaged and customer service does not answer.",
  "Fair price but the manual is incomprehensible.",
  "Excellent value for money, second purchase!"]

# The system prompt is the same for every review: thanks to the KV cache, it
# is only computed once.
let classifier = newChat(model, sampling = greedySampling(), maxTokens = 60, lang = "en", system = """
You classify customer reviews. Answer only in JSON:
{"sentiment": "positive"|"negative"|"neutral", "topic": "delivery"|"product"|"price"|"service"|"other"}""")

var csv = "review;sentiment;topic\n"
for rv in reviews:
  classifier.reset()
  try:
    let j = classifier.askJson(rv)
    csv.add rv.replace(";", ",") & ";" & j{"sentiment"}.getStr("?") & ";" & j{"topic"}.getStr("?") & "\n"
  except ChatError:
    csv.add rv.replace(";", ",") & ";error;error\n"
writeFile("classified_reviews.csv", csv)
echo csv
```

## 7.2 Translating a text file

```nim
# file: translate.nim
## nim c -r translate.nim source.txt French
import std/[os, strutils]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let source = if paramCount() >= 1: readFile(paramStr(1))
             else: "Hello everyone.\n\nThe meeting has been moved to Thursday.\n\nThank you for your understanding."
let language = if paramCount() >= 2: paramStr(2) else: "French"

var p = defaultSampling()
p.temperature = 0.2
let translator = newChat(model, sampling = p, maxTokens = 600, lang = "en",
  system = "You are a professional translator. Translate faithfully into " & language &
           ". Answer only with the translation, without comment.")

var output: seq[string]
for paragraph in source.split("\n\n"):        # paragraph by paragraph
  if paragraph.strip.len == 0: continue
  translator.reset()
  output.add translator.ask(paragraph).text
writeFile("translation.txt", output.join("\n\n"))
echo output.join("\n\n")
```

## 7.3 Questions about your documents (RAG)

*Retrieval-Augmented Generation*: find the relevant passages, then attach them to
the question. The model answers from **your** data, without retraining.

```nim
# file: rag.nim
## nim c -r rag.nim documents_folder
import std/[os, strutils, algorithm, sequtils, sets, unicode]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let folder = if paramCount() >= 1: paramStr(1) else: "documents"

type Passage = object
  source, text: string
  vector: seq[float32]
  words: HashSet[string]

proc wordsOf(s: string): HashSet[string] =
  for w in unicode.toLower(s).split({' ', ',', '.', '\'', '?', '!', ';', ':', '\n', '(', ')'}):
    if w.runeLen >= 4: result.incl w

# 1. Indexing: split into ~600-character passages + vectors
var index: seq[Passage]
if not dirExists(folder):
  createDir(folder)
  writeFile(folder / "hours.txt", "The media library is open from Tuesday to Saturday, 10 am to 6 pm. Annual closure in August.")
  writeFile(folder / "loans.txt", "You can borrow 10 books and 4 DVDs for 3 weeks. Loans can be renewed once online.")
for f in walkFiles(folder / "*.txt"):
  var cur = ""
  for sentence in readFile(f).split(". "):
    cur.add sentence & ". "
    if cur.len > 600:
      index.add Passage(source: f.extractFilename, text: cur.strip)
      cur = ""
  if cur.strip.len > 0: index.add Passage(source: f.extractFilename, text: cur.strip)
for p in index.mitems:
  p.vector = model.embed(p.text)
  p.words = wordsOf(p.text)
echo index.len, " passages indexed"

# 2. Search: hybrid score (semantic + shared words)
proc search(q: string; k = 3): seq[Passage] =
  let vq = model.embed(q)
  let wq = wordsOf(q)
  var scores: seq[(float, int)]
  for i, p in index:
    let lex = (wq * p.words).len.float / max(1, wq.len).float
    scores.add (0.5 * cosineSimilarity(vq, p.vector) + 0.5 * lex, i)
  scores.sort(proc (a, b: (float, int)): int = cmp(b[0], a[0]))
  for j in 0 ..< min(k, scores.len): result.add index[scores[j][1]]

# 3. Answer from the retrieved passages
let chat = newChat(model, nCtx = 8192, maxTokens = 300, sampling = greedySampling(), lang = "en", system =
  "Answer only with the information in the attachments. If they do not " &
  "contain the answer, say \"I don't know\". Cite the source.")
while true:
  stdout.write "\nQuestion (empty to quit): "
  var q: string
  if not stdin.readLine(q) or q.strip.len == 0: break
  let passages = search(q)
  chat.reset()
  echo chat.ask(q, attachments = passages.mapIt(attachText(it.source, it.text))).text
```

For large corpora, compute the index once and store the vectors (for example in
a GGUF file with `GgufWriter.addTensorF32`, see chapter 12).

## 7.4 Summarizing a document longer than the context

The "map-reduce" method: split into pieces that fit in the context, summarize
each one (*map*), then summarize the summaries (*reduce*). Complete program:
[`examples/ex13_resume_long.nim`](nimllm/examples/ex13_resume_long.nim). The key
point is to split by number of **tokens**:

```nim
# file: split_tokens.nim
import std/[os, strutils]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

proc split(tok: Tokenizer; text: string; maxTokens: int): seq[string] =
  ## Pieces of at most `maxTokens` tokens, cut between two paragraphs.
  var cur = ""
  for par in text.split("\n\n"):
    let attempt = if cur.len == 0: par else: cur & "\n\n" & par
    if tok.encode(attempt).len > maxTokens and cur.len > 0:
      result.add cur
      cur = par
    else:
      cur = attempt
  if cur.len > 0: result.add cur

let text = "A fairly long first paragraph.\n\n".repeat(400)
let pieces = split(model.tokenizer, text, 1500)
echo pieces.len, " pieces"
```

## 7.5 Agent with tools

The model chooses, in JSON, a function to call; the program runs it and sends the
result back. Complete, commented program:
[`examples/ex14_outils.nim`](nimllm/examples/ex14_outils.nim). Diagram:

```
question ─► model: {"tool": "calculate", "arguments": {...}}
              │
              ▼
          Nim program runs calculate(...) ─► "69104"
              │
              ▼
          model: "1234 × 56 = 69,104."
```

Tips: describe each tool in one line in the system prompt, enforce JSON with
`format = ofJson`, keep a low temperature, and always check the arguments before
running anything (never let a model run system commands without control).

## 7.6 Generating training data

A large model can produce examples to specialize a small one (chapters 10 and 11):

```nim
# file: generate_data.nim
## Produces a JSONL file of questions/answers on a topic.
import std/[os, json]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let topic = "household waste recycling"
var p = defaultSampling()
p.temperature = 0.9                       # variety
let gen = newChat(model, sampling = p, maxTokens = 250, lang = "en")

var output = open("recycling_data.jsonl", fmWrite)
for i in 1 .. 20:
  gen.reset()
  p.seed = i                              # different seed for each example
  gen.setSampling(p)
  try:
    let j = gen.askJson("Make up a question a resident might ask about " & topic &
                        ", and an accurate, concise answer.",
                        schema = """{"question": str, "answer": str}""")
    let q = j{"question"}.getStr
    let a = j{"answer"}.getStr
    if q.len == 0 or a.len == 0:
      echo i, ". (incomplete, skipped)"
      continue
    let line = %*{"messages": [{"role": "user", "content": q},
                               {"role": "assistant", "content": a}]}
    output.writeLine($line)
    echo i, ". ", q
  except CatchableError:
    echo i, ". (skipped)"
output.close()
```

Always review generated data: a 1B model makes factual mistakes.

## 7.7 A local HTTP service

```nim
# file: server.nim
## Small server: POST /ask with {"question": "..."} -> {"answer": "..."}
## Test: curl -s localhost:8080/ask -d '{"question":"Hello"}'
import std/[os, asynchttpserver, asyncdispatch, json]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let chat = newChat(model, maxTokens = 300, lang = "en")

proc handle(req: Request) {.async, gcsafe.} =
  {.cast(gcsafe).}:
    if req.reqMethod == HttpPost and req.url.path == "/ask":
      try:
        let q = parseJson(req.body)["question"].getStr
        chat.reset()                       # one question = one conversation
        let r = chat.ask(q)
        await req.respond(Http200, $(%*{"answer": r.text, "tokens": r.completionTokens}),
                          newHttpHeaders([("Content-Type", "application/json; charset=utf-8")]))
      except CatchableError as e:
        await req.respond(Http400, $(%*{"error": e.msg}))
    else:
      await req.respond(Http404, "POST /ask")

let server = newAsyncHttpServer()
echo "Listening on http://localhost:8080"
waitFor server.serve(Port(8080), handle)
```

Generation is synchronous: requests are handled one at a time. To serve several
users, create one `Chat` per session (weights are shared) and a queue.

**Next:** **08 — Under the hood**


---

# 08 — Under the hood: tokens, logits, cache, embeddings

Goal: understand and drive the engine directly (without the `Chat` layer). These
concepts are needed for training (chapters 9 to 11).

## 8.1 How an LLM works, on one page

```
"The cat is sleeping"
     │  tokenizer
     ▼
[ 791, 8415, 374, 21811 ]                  token IDs (Llama 3 tokenizer)
     │  embedding table (vocab × dim)
     ▼
4 vectors of 2048 numbers
     │  16 Transformer blocks:
     │    RMS normalization → attention (which previous tokens to look at?)
     │    RMS normalization → feed-forward network (SwiGLU)
     ▼
4 "contextualized" vectors
     │  normalization + output head (dim × vocab)
     ▼
logits: 128,256 scores for the next token
     │  sampler
     ▼
next token ─► appended to the input ─► repeat
```

The **KV cache** stores, for each token already seen, its attention keys and
values: the past is never recomputed, only the new token goes through the
network.

## 8.2 Tokens

```nim
# file: tokens.nim
import std/[os, strutils]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = model.tokenizer

for text in ["Hello", " Hello", "antidisestablishmentarianism", "2024", "🦙", "<|eot_id|>"]:
  let ids = tok.encode(text)                        # parseSpecial = true by default
  var pieces: seq[string]
  for id in ids: pieces.add tok.tokenToPiece(id, renderSpecial = true).escape
  echo alignLeft(text.escape, 30), " -> ", ids, "  ", pieces.join(" | ")

# User text must not create control tokens:
echo tok.encode("<|eot_id|>", parseSpecial = false).len, " tokens (ordinary text)"

echo "Vocabulary: ", tok.vocabSize
echo "Begin/end: ", tok.bosId, " / ", tok.eosId, "; end of turn: ", tok.eogIds
echo "Type: ", tok.kind, " (", tok.pre, ")"
```

Key points:

* the space is part of the token ("Hello" ≠ "␣Hello");
* a rare word is split into several tokens; an emoji into bytes;
* **special tokens** (`<|begin_of_text|>`, `<|eot_id|>`…) structure the
  conversation; `parseSpecial = false` prevents a text from producing them;
* `tok.eogIds` groups the tokens that end an answer.

## 8.3 Chat templates

An "Instruct" model was trained on conversations formatted with tags. For
Llama 3:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

The model then continues the text and ends with `<|eot_id|>`. nimllm detects the
template (`detectTemplate`) from the GGUF metadata:

| Template | Models |
|---|---|
| `tplLlama3` | Llama 3, 3.1, 3.2, 3.3 |
| `tplChatML` | Qwen 2/2.5/3, SmolLM, models created with nimllm |
| `tplMistral` | Mistral, Mixtral (`[INST] … [/INST]`) |
| `tplLlama2` | Llama 2 Chat (`<<SYS>>`) |
| `tplGemma` | Gemma (template only: architecture not supported) |
| `tplPhi3` | Phi-3 (template only) |
| `tplRaw` | base models: "Utilisateur : … Assistant :" (French labels) |

```nim
# file: templates.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let msgs = @[Message(role: roleSystem, content: "Be brief."),
             Message(role: roleUser, content: "Hi!")]
for t in [tplLlama3, tplChatML, tplMistral, tplRaw]:
  echo "=== ", t, "\n", renderPrompt(t, msgs)

# A wrong template badly degrades answers: force it if detection fails
# (model without chat metadata).
let chat = newChat(model, templ = tplLlama3, lang = "en")
echo chat.promptTokens().len, " tokens in the current prompt"
```

## 8.4 Evaluating and reading logits

`LlmContext` is the lower layer: a KV cache and buffers.

```nim
# file: logits.nim
import std/[os, strutils, math]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = model.tokenizer
let ctx = newContext(model, nCtx = 512)

proc nextTokens(text: string) =
  ctx.reset()
  let logits = ctx.eval(tok.encode(text, addBos = true))
  echo "\"", text, "\" →"
  for (id, p) in topTokens(logits, 5):
    echo "    ", (p * 100).formatFloat(ffDecimal, 1).align(5), " %  ", tok.tokenToPiece(id).escape

nextTokens("The Eiffel Tower is located in")
nextTokens("2 + 2 =")
nextTokens("Once upon a")

# Probability of a given continuation: sum of the log-probabilities of each token
proc logProb(start, continuation: string): float =
  ctx.reset()
  var logits = ctx.eval(tok.encode(start, addBos = true))
  for id in tok.encode(continuation):
    var p = logits
    softmaxInPlace(p)
    result += ln(p[id].float)
    logits = ctx.eval([id])

echo "log P(\"Paris\") = ", logProb("The capital of France is", " Paris").formatFloat(ffDecimal, 2)
echo "log P(\"Lyon\")  = ", logProb("The capital of France is", " Lyon").formatFloat(ffDecimal, 2)
```

Useful context functions:

| Function | Role |
|---|---|
| `newContext(model, nCtx, nBatch)` | creates a context (cache of `nCtx` tokens) |
| `ctx.eval(tokens)` | appends tokens, returns the logits of the last one |
| `ctx.eval(tokens, allLogits = true)` | logits of every position (concatenated) |
| `ctx.evalPrompt(tokens)` | like `eval`, reusing the prefix already in the cache |
| `ctx.tokens` / `ctx.nPast` | cache contents / size |
| `ctx.truncate(n)` / `ctx.reset()` | forgets the end / everything |
| `ctx.lastHidden` | final hidden states of the last batch |

## 8.5 Writing your own generation loop

```nim
# file: loop.nim
## Manual generation with a filter: digits are forbidden.
import std/[os, strutils, sequtils]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let tok = model.tokenizer
let ctx = newContext(model, nCtx = 1024)
let s = newSampler(defaultSampling())

# Preparing the prompt with the template
let prompt = encodeMessages(tok, detectTemplate(tok),
  @[Message(role: roleUser, content: "How old is the Eiffel Tower?")])
var logits = ctx.eval(prompt)

var forbidden: seq[int]
for id in 0 ..< tok.vocabSize:
  if tok.tokenToPiece(id).anyIt(it.isDigit): forbidden.add id

for i in 0 ..< 80:
  for id in forbidden: logits[id] = -Inf     # the model must write numbers in words
  let id = s.sample(logits)
  if id in tok.eogIds: break
  s.accept(id)                               # for repetition penalties
  stdout.write tok.tokenToPiece(id)
  stdout.flushFile()
  logits = ctx.eval([id])
echo ""
```

## 8.6 Raw completion and `generate`

`complete` continues a text without a template: useful for base (non-"Instruct")
models or to force the beginning of an answer.

```nim
# file: completion.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let ctx = newContext(model, nCtx = 1024)
var o = defaultOptions()
o.maxTokens = 60
o.stop = @["\n\n"]
echo ctx.complete("Pancake recipe\nIngredients:\n-", o).text

# Forcing the beginning of an assistant's answer:
let tok = model.tokenizer
var p = encodeMessages(tok, detectTemplate(tok),
  @[Message(role: roleUser, content: "Name three planets.")])
p.add tok.encode("Here are three planets, in alphabetical order:", parseSpecial = false)
ctx.reset()
echo ctx.generate(p, o).text
```

## 8.7 Embeddings and similarity

`model.embed(text)` returns a normalized vector (average of the final hidden
states): two similar texts have a high cosine.

```nim
# file: similarity.nim
import std/[os, strutils]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let sentences = ["The cat is sleeping on the couch.", "A feline is napping on the sofa.",
                 "The Paris stock exchange fell.", "Financial markets are going down."]
var vectors: seq[seq[float32]]
for s in sentences: vectors.add model.embed(s)
for i in 0 ..< sentences.len:
  for j in i+1 ..< sentences.len:
    echo cosineSimilarity(vectors[i], vectors[j]).formatFloat(ffDecimal, 3), "  ",
         sentences[i], " ↔ ", sentences[j]
```

Embeddings from a generative model are less discriminating than those of a
specialized model; combine them with a lexical score (see 7.3).

## 8.8 Perplexity

Perplexity measures how much a text "surprises" the model (lower = more natural
to it). It is used to compare models, to measure the quality loss due to
quantization, or to detect abnormal text.

```nim
# file: perplexity.nim
import std/[os, strutils]
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
for t in ["The little cat drinks milk.", "Milk drinks cat little the.",
          "Colorless green ideas sleep furiously."]:
  echo model.perplexity(t).formatFloat(ffDecimal, 1).align(9), "  ", t
```

## 8.9 Engine architecture

| Element | Implementation |
|---|---|
| Weights | mapped from the file (`memfiles`), never copied |
| Matrix-vector product | activations quantized to int8 per block + integer dot products (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0), as in ggml; other formats: the row is dequantized, then a floating-point product |
| Parallelism | custom thread pool (`parallel.nim`): matrix rows, attention heads |
| Attention | grouped multi-head (GQA), causal, float32 KV cache |
| Position | RoPE (adjacent pairs for llama, halves for qwen) + Llama 3 factors (`rope_freqs`) |
| Prompt | processed in batches of `nBatch` tokens (32 by default) |
| LoRA | applied on the fly (`applyLora`): y = W·x + scale·B·(A·x) |

`exactMatmul = true` disables activation quantization (exact float32 computation,
slower), useful for comparing results.

**Next:** **09 — Machine learning basics**


---

# 09 — Machine learning basics

Goal: understand how a neural network learns, using nimllm's `autograd` module.
These basics are then used to fine-tune (chapter 10) and build (chapter 11)
language models. No GGUF model is needed here.

## 9.1 The principle

Learning = adjusting **parameters** to lower a **loss** (the gap between what the
model predicts and what it should predict):

```
repeat:
  1. forward pass     : compute the loss from the data
  2. backpropagation  : compute the gradient of the loss for each parameter
  3. update           : move each parameter a little in the direction that reduces the loss
```

The **gradient** tells in which direction (and how strongly) to change each
parameter. **Automatic differentiation** computes it for you: you only write the
computation of the loss.

## 9.2 Tensors

A `Tensor` holds `float32` values (`data`), a shape (`shape`) and, if it is
trainable, its gradient (`grad`). Matrices are stored row by row: an `[N, C]`
tensor contains N rows of C values.

```nim
# file: tensors.nim
import std/random
import nimllm

var rng = initRand(1)
let a = fromSeq(@[1'f32, 2, 3, 4, 5, 6], [2, 3])     # 2×3 matrix
let z = newTensor([2, 3])                              # zeros
let u = full([3], 1.0)                                 # vector of ones
let w = randn([4, 3], std = 0.1, rng)                  # random N(0, 0.1²), trainable
echo a                                                 # Tensor@[2, 3] [1.0, 2.0, ... 6.0]
echo a.rows, " rows × ", a.cols, " columns, ", a.numel, " values"
echo w.requiresGrad, " ", z.requiresGrad               # true false

let total = a + a                 # element-wise
let product = a * a
let broadcast = a + u             # a [C] vector is added to every row
let lin = linear(a, w)            # a·wᵀ: [2,3]·[3,4] -> [2,4]
echo total, "\n", product, "\n", broadcast, "\n", lin.shape
```

Available operations:

| Category | Functions |
|---|---|
| Arithmetic | `+`, `-`, `*` (element-wise or broadcast of a row vector), `scale`, `sum`, `mean` |
| Algebra | `linear(x, w, bias)` (x·wᵀ+b), `matmul(a, b)`, `concatCols` |
| Activations | `relu`, `silu`, `gelu`, `sigmoid`, `tanhT`, `softmax` |
| Normalization | `rmsnorm(x, gain)` |
| Language | `embedding(table, ids)`, `rope`, `causalAttention`, `crossEntropy` |
| Losses | `crossEntropy(logits, targets)`, `mseLoss(prediction, target)` |
| Misc | `reshape`, `detach`, `dropout`, `noGrad:` |

## 9.3 Automatic gradients

```nim
# file: gradient.nim
import nimllm

# f(x, y) = sum(x * y + x)   =>  df/dx = y + 1; df/dy = x
let x = fromSeq(@[1'f32, 2, 3], [3], requiresGrad = true)
let y = fromSeq(@[4'f32, 5, 6], [3], requiresGrad = true)
let f = sum(x * y + x)
backward(f)                       # fills x.grad and y.grad
echo "f = ", f.item               # 1*4+1 + 2*5+2 + 3*6+3 = 38
echo "df/dx = ", x.grad           # @[5.0, 6.0, 7.0]
echo "df/dy = ", y.grad           # @[1.0, 2.0, 3.0]

# Gradients accumulate: reset them before each step.
x.zeroGrad(); y.zeroGrad()

# During inference, disable graph construction (faster, less memory):
noGrad:
  let g = sum(x * y)
  echo g.item, " (no graph: ", g.requiresGrad, ")"
```

## 9.4 First training: a linear regression

We look for `w` and `b` such that `y ≈ w·x + b`, from noisy examples.

```nim
# file: regression.nim
import std/[random, strutils]
import nimllm

var rng = initRand(42)
# Data: y = 2.5·x − 1 + noise
var xs, ys: seq[float32]
for i in 0 ..< 100:
  let x = rng.rand(4.0) - 2.0
  xs.add float32(x)
  ys.add float32(2.5 * x - 1.0 + rng.gauss() * 0.1)
let X = fromSeq(xs, [100, 1])          # 100 examples, 1 feature
let Y = fromSeq(ys, [100, 1])

let w = param([1, 1], 0.1, rng, "w")   # trainable parameters
let b = full([1], 0, true, "b")
let opt = newSGD(@[w, b], lr = 0.05, momentum = 0.9)

for step in 1 .. 200:
  opt.zeroGrad()                        # 0. zero the gradients
  let prediction = linear(X, w, b)      # 1. forward pass
  let loss = mseLoss(prediction, Y)
  backward(loss)                        # 2. backpropagation
  opt.update()                          # 3. update
  if step mod 40 == 0:
    echo "step ", step, "  loss ", loss.item.formatFloat(ffDecimal, 5),
         "  w = ", w.data[0].formatFloat(ffDecimal, 3), "  b = ", b.data[0].formatFloat(ffDecimal, 3)
```

## 9.5 A real neural network: classifying points

Two clouds of points wound in a spiral cannot be separated by a straight line:
non-linear hidden layers are needed.

```nim
# file: spirals.nim
import std/[random, math, strutils]
import nimllm

var rng = initRand(7)
# 1. Data: two spirals of 100 points
var pts: seq[float32]
var classes: seq[int]
for c in 0 .. 1:
  for i in 0 ..< 100:
    let r = i.float / 100.0
    let t = c.float * PI + r * 4.0 + rng.gauss() * 0.15
    pts.add float32(r * cos(t)); pts.add float32(r * sin(t))
    classes.add c
let X = fromSeq(pts, [200, 2])

# 2. Model: 2 -> 32 -> 32 -> 2
let w1 = param([32, 2], 1.0, rng);  let b1 = full([32], 0, true)
let w2 = param([32, 32], 0.2, rng); let b2 = full([32], 0, true)
let w3 = param([2, 32], 0.2, rng);  let b3 = full([2], 0, true)
let params = @[w1, b1, w2, b2, w3, b3]

proc net(x: Tensor): Tensor =
  let h1 = relu(linear(x, w1, b1))
  let h2 = relu(linear(h1, w2, b2))
  linear(h2, w3, b3)                      # logits of the 2 classes

# 3. Training with AdamW (the optimizer used for LLMs)
let opt = newAdamW(params, lr = 0.01, weightDecay = 0.0)
for step in 1 .. 1000:
  opt.zeroGrad()
  let loss = crossEntropy(net(X), classes)
  backward(loss)
  discard clipGradNorm(params, 1.0)       # avoids gradient "explosions"
  opt.update()
  if step mod 200 == 0:
    var correct = 0
    noGrad:
      let p = net(X)
      for i in 0 ..< 200:
        let pred = if p.data[2*i+1] > p.data[2*i]: 1 else: 0
        if pred == classes[i]: inc correct
    echo "step ", step, "  loss ", loss.item.formatFloat(ffDecimal, 4),
         "  accuracy ", correct div 2, " %"
```

## 9.6 What happens inside an LLM

A language model is just a bigger network, with the same building blocks:

```nim
# file: mini_transformer.nim
## A Transformer block written by hand with autograd operations.
import std/[random, math]
import nimllm

var rng = initRand(3)
const V = 50      # vocabulary size
const D = 32      # vector dimension
const H = 4       # attention heads
const T = 8       # sequence length

let emb = param([V, D], 0.02, rng)
let normA = full([D], 1, true)
let wq = param([D, D], 0.02, rng); let wk = param([D, D], 0.02, rng)
let wv = param([D, D], 0.02, rng); let wo = param([D, D], 0.02, rng)
let normF = full([D], 1, true)
let w1 = param([4*D, D], 0.02, rng); let w2 = param([D, 4*D], 0.02, rng)
let normS = full([D], 1, true)

var invFreq: seq[float32]
for i in 0 ..< (D div H) div 2: invFreq.add float32(pow(10000.0, -2.0 * i.float / (D div H).float))
var positions: seq[int]
for t in 0 ..< T: positions.add t

proc forwardPass(ids: seq[int]): Tensor =
  var x = embedding(emb, ids)                                  # [T, D]
  let h = rmsnorm(x, normA)
  let q = rope(linear(h, wq), H, D div H, positions, invFreq)
  let k = rope(linear(h, wk), H, D div H, positions, invFreq)
  let v = linear(h, wv)
  x = x + linear(causalAttention(q, k, v, 1, T, H, H, D div H), wo)   # attention + residual
  x = x + linear(silu(linear(rmsnorm(x, normF), w1)), w2)              # feed-forward + residual
  linear(rmsnorm(x, normS), emb)                                       # logits [T, V] (tied weights)

# Toy task: learn the sequence shifted by one (predict the next token)
let params = @[emb, normA, wq, wk, wv, wo, normF, w1, w2, normS]
let opt = newAdamW(params, lr = 3e-3)
for step in 1 .. 300:
  var ids: seq[int]
  for t in 0 .. T: ids.add (t * 3 + step) mod V          # arithmetic sequence
  opt.zeroGrad()
  let loss = crossEntropy(forwardPass(ids[0 ..< T]), ids[1 .. T])
  backward(loss)
  opt.update()
  if step mod 100 == 0: echo "step ", step, " loss ", loss.item
```

This is exactly the structure of `Transformer` (`nn` module), which adds several
blocks, grouped attention, GGUF export, LoRA, etc.

## 9.7 Writing your own operation

Any function can become differentiable: compute the result, then register the
backpropagation function with `makeNode`. Check it with `gradCheck` (finite
differences).

```nim
# file: operation.nim
import std/[random, math]
import nimllm

proc softplus(a: Tensor): Tensor =
  ## y = ln(1 + eˣ); dy/dx = sigmoid(x)
  result = newTensor(a.shape)
  for i in 0 ..< a.numel: result.data[i] = ln(1 + exp(a.data[i]))
  if needsGrad(a):
    let r = result
    result.makeNode(@[a], proc () =
      a.ensureGrad()
      for i in 0 ..< a.numel:
        a.grad[i] += r.grad[i] * (1 / (1 + exp(-a.data[i]))))

var rng = initRand(5)
let x = randn([4, 5], 1.0, rng)
echo "relative error: ", gradCheck(proc (): Tensor = sum(softplus(x) * x), x)   # ~1e-3 or less
```

## 9.8 Training vocabulary

| Term | Meaning | Typical values |
|---|---|---|
| learning rate (`lr`) | size of the update steps | 1e-4 to 3e-3 (AdamW) |
| *warmup* | gradual increase of the rate at the start | 1 to 10% of the steps |
| cosine decay | gradual decrease of the rate | down to `minLr` ≈ lr/10 |
| *weight decay* | pulls weights toward 0 (regularization) | 0 to 0.1 |
| clipping (`gradClip`) | limits the gradient norm | 1.0 |
| batch (`batchSize`) | examples processed together | 4 to 64 |
| epoch | one pass over all the data | — |
| overfitting | the model recites the training data but generalizes poorly | watch it with a validation set |

**Next:** **10 — Fine-tuning with LoRA**


---

# 10 — Fine-tuning an existing model with LoRA

Goal: specialize a model such as Llama 3.2 on **your** data (domain vocabulary,
style, answer format, company knowledge), on an ordinary CPU.

## 10.1 Why LoRA

Retraining all the weights of Llama 3.2 1B would take ~20 GB of memory (weights +
gradients + AdamW states in float32). **LoRA** (*Low-Rank Adaptation*) freezes the
original weights and adds, next to some matrices W, two small matrices A (r ×
input) and B (output × r):

```
y = W·x  +  (alpha / r) · B·(A·x)
    frozen     learned (r = 8: ~0.5% of the parameters)
```

In nimllm, the frozen weights stay **quantized and mapped** from the GGUF: the
memory needed is that of inference plus the activations of the current batch
(about 10 MB per batch token for Llama 3.2 1B).

| | Full fine-tuning (`tmFull`) | LoRA (`tmLora`) |
|---|---|---|
| Parameters learned | all | 0.1 to 2% |
| Memory (Llama 3.2 1B) | ~20 GB | ~2 GB (128-token batch) to ~6 GB (512 tokens) |
| File produced | full model | adapter of a few MB |
| Recommended use | small models (< 50 M) | existing models (1B, 3B, 8B) |

**What LoRA does well**: impose a style, a format, a tone, a vocabulary, learn
standard answers. **What it does less well**: add a lot of new knowledge (prefer
RAG then, chapter 7.3, possibly combined).

## 10.2 Preparing the data

A **JSONL** file: one conversation per line. Three formats are accepted:

```json
{"messages": [{"role": "user", "content": "Are you open on Mondays?"}, {"role": "assistant", "content": "No, we are closed on Mondays."}]}
{"prompt": "Where are you located?", "response": "At 12 rue des Lilas, in Nantes."}
{"instruction": "Give the email address.", "input": "", "output": "contact@boulangerie-dupont.fr"}
```

Tips:

* **quality > quantity**: 50 to 500 carefully written examples are often enough;
* vary the wording of the questions;
* use exactly the answer style you want (length, tone, format);
* keep 5 to 10% of the examples for **validation**;
* only the assistant's answers are learned (the loss ignores the rest).

Example provided (in French):
[`examples/donnees/faq_boulangerie.jsonl`](nimllm/examples/donnees/faq_boulangerie.jsonl).

## 10.3 The complete program

```nim
# file: lora_bakery.nim
## nim c -r lora_bakery.nim model.gguf data.jsonl
import std/[os, math, strutils]
import nimllm

let path = if paramCount() >= 1: paramStr(1)
           else: getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let dataFile = if paramCount() >= 2: paramStr(2) else: "faq_boulangerie.jsonl"
let system = "You are the assistant of the Dupont bakery."

# 1. Model in LoRA mode
var lc = defaultLora()           # rank 8, alpha 16, layers q,k,v,o
lc.rank = 16
lc.alpha = 32
lc.targets = @["q", "k", "v", "o", "gate", "up", "down"]
let m = loadForTraining(path, tmLora, lc)
echo "Parameters learned: ", m.parameterCount

# 2. Data (same template as the model, same system prompt as in use)
let all = loadChatJsonl(dataFile, m.tokenizer, detectTemplate(m.tokenizer), system = system)
let (train, valid) = all.split(0.1)
var longest = 0
for ex in all.examples: longest = max(longest, ex.tokens.len)

# 3. Settings
var tc = defaultTrainConfig()
tc.steps = 200
tc.batchSize = 1              # 1 sequence at a time: minimal memory
tc.gradAccum = 8              # ... but gradients accumulated over 8 examples
tc.seqLen = min(longest, 256)
tc.lr = 1e-3
tc.warmup = 20
tc.evalEvery = 50
tc.logEvery = 10
tc.saveEvery = 100
tc.savePath = "bakery"        # -> bakery.lora.gguf (+ optimizer state)

# 4. Training
echo "Initial validation loss: ", m.evaluate(valid, 2, tc.batchSize, tc.seqLen).formatFloat(ffDecimal, 3)
discard m.train(train, tc, valData = valid)
m.saveLora("bakery.lora.gguf")

# 5. Immediate test
let base = loadModel(path)
base.applyLora("bakery.lora.gguf")
let chat = newChat(base, system = system, sampling = greedySampling(), maxTokens = 80, lang = "en")
for q in ["Are you open on Monday?", "How much is a baguette?"]:
  chat.reset()
  echo q, "\n→ ", chat.ask(q).text
```

Commented version in the examples: [`examples/ex18_lora.nim`](nimllm/examples/ex18_lora.nim).

## 10.4 Reading the curves

The log prints lines like this one (the library's log labels are in French:
*étape* = step, *perte* = loss, *val* = validation loss):

```
étape    50 | perte 1.2345 | val 1.3012 | lr 9.51e-04 | ‖g‖ 0.82 | 410 tok/s
```

* **loss** (training) should go down; close to 0 = the model recites the examples;
* **val** (validation): if it **goes back up** while the loss goes down, that is
  overfitting → fewer steps, lower rank, more data;
* **‖g‖**: gradient norm; very large, unstable values → lower `lr`;
* a loss that stalls from the start → raise `lr` (LoRA handles 1e-4 to 3e-3).

## 10.5 Choosing hyperparameters

| Parameter | Effect | Starting point |
|---|---|---|
| `rank` | adapter capacity | 8 (style), 16–32 (content) |
| `alpha` | strength of the adaptation | 2 × rank |
| `targets` | adapted layers | `q,k,v,o` (light); + `gate,up,down` (more effective) |
| `lr` | learning speed | 1e-3 |
| `steps` | duration | 2 to 5 passes over the data |
| `seqLen` | max length of the examples | length of the longest example |

Number of passes over the data ≈ `steps × batchSize × gradAccum / number of examples`.

## 10.6 Using the adapter

Two options:

```nim
# file: use_lora.nim
import std/os
import nimllm

let path = getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

# a) On the fly: the base model is unchanged, you can switch adapters.
let m = loadModel(path)
m.applyLora("bakery.lora.gguf")
echo newChat(m, lang = "en").ask("What are your opening hours?").text
m.removeLora()                                   # back to the original model

# b) Merging: a new self-contained GGUF (no inference overhead).
#    Modified tensors are re-quantized to Q8_0 (or gtQ4_K, gtF16...).
mergeLora(path, "bakery.lora.gguf", "llama-bakery.gguf", outType = gtQ8_0)
let merged = loadModel("llama-bakery.gguf")
echo newChat(merged, lang = "en").ask("What are your opening hours?").text
```

The merged GGUF is a standard file: usable with llama.cpp, LM Studio or Ollama
(`FROM ./llama-bakery.gguf` in a `Modelfile`).

## 10.7 Resuming or continuing a fine-tuning

`saveEvery`/`saveCheckpoint` write the adapter and the optimizer state. To
continue later:

```nim
# file: resume_lora.nim
import std/os
import nimllm

let path = getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
var lc = defaultLora()
lc.rank = 16; lc.alpha = 32
lc.targets = @["q", "k", "v", "o", "gate", "up", "down"]
let m = loadForTraining(path, tmLora, lc)

# reload the saved A and B matrices
let g = openGguf("bakery.lora.gguf")
for p in m.parameters:
  if p.name in g.tensors: p.data = g.tensorF32(p.name)
g.close()

let opt = newAdamW(m.parameters, lr = 5e-4)
if fileExists("bakery.optim.gguf"): opt.loadState("bakery.optim.gguf")
let ds = loadChatJsonl("faq_boulangerie.jsonl", m.tokenizer, detectTemplate(m.tokenizer),
                       system = "You are the assistant of the Dupont bakery.")
var tc = defaultTrainConfig()
tc.steps = 50; tc.batchSize = 1; tc.gradAccum = 8; tc.seqLen = 256; tc.lr = 5e-4; tc.evalEvery = 0
discard m.train(ds, tc, opt = opt)
m.saveLora("bakery.lora.gguf")
```

## 10.8 Duration on a CPU

A LoRA training step costs about 3 times the processing of the same tokens during
inference. Reference measurement for a model the size of Llama 3.2 1B (Q4_K_M),
on a **2-core** CPU: ~30 s per step for a 128-token batch (≈ 4 tokens learned
per second), memory peak ≈ 2.2 GB. The time is proportional to the number of
tokens in the batch and decreases with the number of cores.

Example: 200 dialogues of 100 tokens, 3 passes = 60,000 tokens ≈ 4 h on 2 cores,
≈ 1 h on 8 cores. For quick trials, start with few examples and few steps, and
validate your whole pipeline on the small model from chapter 11 (a few seconds).

## 10.9 Full fine-tuning of a small model

For a model of a few million parameters (like those in chapter 11), all the
weights can be trained:

```nim
# file: full_finetune.nim
import nimllm

let m = loadForTraining("mini-assistant-f32.gguf", tmFull)   # all weights in float32
echo m.parameterCount, " trainable parameters"
var convs = @[@[Message(role: roleUser, content: "What is the capital of Peru?"),
                Message(role: roleAssistant, content: "The capital of Peru is Lima.")]]
let ds = newChatDataset(m.tokenizer, tplChatML, convs)
var tc = defaultTrainConfig()
tc.steps = 100; tc.batchSize = 4; tc.seqLen = 48; tc.lr = 1e-3; tc.evalEvery = 0
discard m.train(ds, tc)
m.saveGguf("mini-assistant-v2.gguf")
```

**Next:** **11 — Building a new model**


---

# 11 — Building a new model

Goal: design and train a language model **from scratch**, then use it exactly
like Llama 3.2 (and even in llama.cpp or Ollama).

The complete program for this chapter is
[`examples/ex17_creer_modele.nim`](nimllm/examples/ex17_creer_modele.nim): in about
two minutes on a CPU, it produces a mini-assistant that answers questions about
geography, arithmetic and the calendar (in French).

## 11.1 The steps

```
1. data           raw text (pre-training) + dialogues (fine-tuning)
2. tokenizer      trainBpe  ─────────────►  vocabulary
3. architecture   newModelConfig + newTransformer
4. pre-training   : predict the next word on raw text
5. chat fine-tuning (SFT): learn to answer
6. evaluation     validation loss, perplexity, trials
7. export         saveGguf (+ quantizeModel)
8. use            loadModel + newChat, like any other model
```

Orders of magnitude:

| Model | Parameters | Training data | Hardware |
|---|---|---|---|
| this chapter's example | 1 M | a few KB | 1 CPU, minutes |
| small specialized model | 10–50 M | 10–500 MB of text | CPU, hours to days |
| Llama 3.2 1B | 1.2 B | ~9 trillion tokens | thousands of GPUs |

A model built here only knows what is in your data: that is ideal for a narrow
domain (device commands, closed FAQ, formal language, game…).

## 11.2 The tokenizer

```nim
# file: step_tokenizer.nim
import nimllm

let texts = @["The cat sleeps. The dog runs. The cat eats.",
              "An apple is red. A banana is yellow."]
let tok = trainBpe(texts,
  vocabSize = 400,                       # 256 bytes + merges + specials
  specials = @["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"],
  minFreq = 2)                           # merge only if the pair appears ≥ 2 times
echo tok.vocabSize, " tokens; \"The cat sleeps.\" -> ", tok.encode("The cat sleeps.")
echo "Template stored: ", tok.chatTemplate.len > 0      # ChatML (llama.cpp will read it)
```

Vocabulary size: 256 bytes minimum; 1,000–8,000 for a small model; Llama 3 uses
128,256 tokens. A larger vocabulary shortens sequences but enlarges the embedding
table.

The `<|im_start|>`/`<|im_end|>` tokens enable the **ChatML** template: `newChat`
detects it automatically.

## 11.3 The architecture

```nim
# file: step_architecture.nim
import nimllm

let tok = byteLevelTokenizer()            # byte-by-byte tokenizer (no training)
let cfg = newModelConfig(
  vocab = tok.vocabSize,
  dim = 256,          # vector width (embedding_length)
  layers = 6,         # number of Transformer blocks
  heads = 8,          # attention heads (dim divisible by heads)
  kvHeads = 2,        # key/value heads (GQA; heads a multiple of kvHeads)
  hidden = 0,         # feed-forward size (0 = ≈ 8/3 × dim)
  ctx = 256,          # maximum context
  ropeBase = 10000,
  tied = true,        # output head shared with the embeddings
  name = "my-model")
let m = newTransformer(cfg, tok, seed = 1)
echo m.parameterCount, " parameters"
echo "head dimension: ", cfg.headDim, ", feed-forward: ", cfg.hidden
```

Rules of thumb:

* `dim / heads` = 32 to 128;
* for Q4_K/Q6_K export, choose `dim` and `hidden` as multiples of 256 (otherwise
  those tensors are stored in F16, which is still valid);
* doubling `dim` multiplies the cost by ~4; doubling `layers` by ~2.

## 11.4 Pre-training

The model learns the language by predicting the next token on raw text.

```nim
# file: step_pretrain.nim
import std/[os, strutils]
import nimllm

# Corpus: one or more text files (separate documents with 3 line breaks)
let text = if paramCount() >= 1: readFile(paramStr(1))
           else: "Once upon a time there was a small village by the sea. ".repeat(300)
let tok = trainBpe([text], vocabSize = 1000)
let m = newTransformer(newModelConfig(tok.vocabSize, dim = 128, layers = 4, heads = 4, ctx = 128), tok)

let (train, valid) = newTextDataset(tok, text).split(0.05)
echo train.len, " training tokens"

var tc = defaultTrainConfig()
tc.steps = 300
tc.batchSize = 8
tc.seqLen = 128
tc.lr = 1e-3            # 1e-3 to 3e-3 for a small model; 3e-4 for a larger one
tc.minLr = 1e-4
tc.warmup = 50
tc.weightDecay = 0.1
tc.evalEvery = 100
tc.sampleEvery = 100    # prints a sample of generated text
tc.samplePrompt = "Once upon"
discard m.train(train, tc, valData = valid)
m.saveGguf("pretrain.gguf")
```

How to read the loss (cross-entropy, in nats): `ln(vocab)` at the start (6.9 for
1,000 tokens); below 2 the text becomes readable; below 1 on a small corpus, the
model starts reciting.

## 11.5 Chat fine-tuning (SFT)

Show the model conversations: it learns to answer (only the answers count in the
loss).

```nim
# file: step_sft.nim
import nimllm

let m = loadForTraining("pretrain.gguf", tmFull)      # starts from the pre-trained model
var dialogues: seq[seq[Message]]
for (q, a) in [("Hello!", "Hello! What can I do for you?"),
               ("Where is the village?", "The village is by the sea.")]:
  dialogues.add @[Message(role: roleUser, content: q),
                  Message(role: roleAssistant, content: a)]
let ds = newChatDataset(m.tokenizer, tplChatML, dialogues)
# (or: loadChatJsonl("dialogues.jsonl", m.tokenizer, tplChatML))

var tc = defaultTrainConfig()
tc.steps = 300; tc.batchSize = 8; tc.seqLen = 64; tc.lr = 1e-3; tc.evalEvery = 0
discard m.train(ds, tc)
m.saveGguf("assistant.gguf")
```

To keep the model's general knowledge, mix a few pre-training texts into the
dialogues, or use a lower learning rate.

## 11.6 Evaluating

```nim
# file: step_evaluation.nim
import std/[math, strutils]
import nimllm

let lm = loadModel("assistant.gguf")
echo "Perplexity: ", lm.perplexity("The village is by the sea.").formatFloat(ffDecimal, 2)
let chat = newChat(lm, sampling = greedySampling(), maxTokens = 40, nCtx = 128)
for q in ["Hello!", "Where is the village?", "What time is it?"]:
  chat.reset()
  echo q, " -> ", chat.ask(q).text
```

Always test with questions that are **absent** from the training data: it is the
only way to measure generalization.

## 11.7 Exporting, quantizing, sharing

```nim
# file: step_export.nim
import nimllm

let m = loadForTraining("assistant.gguf", tmFull)
m.saveGguf("assistant-f16.gguf", gtF16)          # half the size, no noticeable loss
quantizeModel("assistant.gguf", "assistant-q8.gguf", gtQ8_0)
quantizeModel("assistant.gguf", "assistant-q4k.gguf", gtQ4_K)
```

The file contains the architecture (`llama`), the tokenizer (BPE, Llama 3
pre-tokenization), the ChatML template and the weights: it works with nimllm, with
**llama.cpp** (`llama-cli -m assistant-q8.gguf`), and with **Ollama**:

```
# Modelfile
FROM ./assistant-q8.gguf
```

```sh
ollama create my-assistant -f Modelfile
ollama run my-assistant
```

## 11.8 Tips for going further

* **Data**: the number one factor. Clean, deduplicate, balance.
* **Scale**: for a fixed compute budget, a slightly smaller model trained on more
  data is better (about 20 tokens of data per parameter).
* **Learning rate**: if the loss explodes or becomes `nan`, divide `lr` by 3; if
  it decreases very slowly, multiply it by 2.
* **Context**: `seqLen` ≤ `ctx`; longer sequences cost more (attention is
  quadratic).
* **Resuming**: `saveEvery` + `saveCheckpoint` + `Optimizer.loadState`
  ([`examples/ex20_reprise_entrainement.nim`](nimllm/examples/ex20_reprise_entrainement.nim)).
* **Distillation**: generate dialogues with a large model (chapter 7.6) to train
  a small specialized model.

**Next:** **12 — GGUF and quantization**


---

# 12 — GGUF and quantization

Goal: understand, inspect and produce model files.

## 12.1 Structure of a GGUF file

```
┌──────────────────────────────┐
│ "GGUF", version, counts      │
├──────────────────────────────┤
│ key/value metadata           │  general.architecture = "llama"
│                              │  llama.block_count = 16
│                              │  tokenizer.ggml.tokens = [...]
│                              │  tokenizer.chat_template = "..."
├──────────────────────────────┤
│ tensor descriptions          │  name, dimensions, type, offset
├──────────────────────────────┤
│ data (32-byte aligned)       │  quantized weights
└──────────────────────────────┘
```

Tensor names (llama.cpp convention): `token_embd.weight`,
`blk.N.attn_norm.weight`, `blk.N.attn_q/k/v/output.weight`,
`blk.N.ffn_norm.weight`, `blk.N.ffn_gate/up/down.weight`, `output_norm.weight`,
`output.weight` (absent if weights are tied), `rope_freqs.weight` (Llama 3.1+).

## 12.2 Inspecting

```nim
# file: inspect.nim
import std/[os, strutils, tables]
import nimllm

let g = openGguf(if paramCount() >= 1: paramStr(1)
                 else: getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
echo g.describe(maxTensors = 12)

let arch = g.getStr("general.architecture")
echo "Layers: ", g.getInt(arch & ".block_count")
echo "Context: ", g.getInt(arch & ".context_length")
echo "Vocabulary: ", g.getStrArray("tokenizer.ggml.tokens").len
echo "Template:\n", g.getStr("tokenizer.chat_template")[0 ..< min(300, g.getStr("tokenizer.chat_template").len)]

var byType = initCountTable[GgmlType]()
for name, t in g.tensors: byType.inc(t.typ, t.byteSize)
for typ, bytes in byType:
  echo align($typ, 8), ": ", bytes div (1024*1024), " MiB"
g.close()
```

## 12.3 Weight formats

| Type | Bits/weight | Block | Quality | Use |
|---|---|---|---|---|
| F32 | 32 | 1 | reference | training, small tensors |
| F16 / BF16 | 16 | 1 | near-identical | export without noticeable loss |
| Q8_0 | 8.5 | 32 | excellent | safe default, LoRA merging |
| Q6_K | 6.56 | 256 | very good | output heads |
| Q5_K | 5.5 | 256 | good | compromise |
| Q4_K | 4.5 | 256 | fair | **the most common** (Q4_K_M) |
| Q5_0 / Q5_1 / Q4_0 / Q4_1 | 5.5 / 6 / 4.5 / 5 | 32 | fair | older formats |
| Q3_K / Q2_K | 3.4 / 2.6 | 256 | degraded | read-only in nimllm |

"Q4_K_M" means a *mix*: most matrices in Q4_K, some (output, `attn_v`,
`ffn_down`) in Q6_K.

How a Q8_0 block works: 32 values → one scale `d` (float16) + 32 integers from
−127 to 127; value ≈ d × q. The "K" formats group 256 values into sub-blocks
whose scales are themselves quantized.

```nim
# file: formats.nim
import std/[math, strutils]
import nimllm

var v = newSeq[float32](512)
for i in 0 ..< v.len: v[i] = float32(sin(i.float * 0.1) + 0.3 * cos(i.float * 0.7))
for t in [gtF16, gtBF16, gtQ8_0, gtQ6_K, gtQ5_K, gtQ5_0, gtQ4_K, gtQ4_0]:
  let bytes = quantize(t, v)                         # float32 -> blocks
  let back = dequantRow(t, unsafeAddr bytes[0], v.len)      # blocks -> float32
  var err = 0.0
  for i in 0 ..< v.len: err += (v[i] - back[i]).float ^ 2
  echo align($t, 7), ": ", align($bytes.len, 5), " bytes, RMS error ",
       sqrt(err / v.len.float).formatFloat(ffScientific, 2)
```

## 12.4 Quantizing a model

```nim
# file: quantize.nim
## nim c -r quantize.nim input.gguf output.gguf q4_k
import std/os
import nimllm

let input = paramStr(1)
let output = paramStr(2)
let t = if paramCount() >= 3: parseGgmlType(paramStr(3)) else: gtQ8_0
quantizeModel(input, output, t, keepOutput = true)
# keepOutput: embeddings and output head stay in Q8_0 (better quality)
echo getFileSize(input) div (1024*1024), " MiB -> ", getFileSize(output) div (1024*1024), " MiB"
echo "perplexity before: ", loadModel(input).perplexity("A representative test text.")
echo "perplexity after: ", loadModel(output).perplexity("A representative test text.")
```

Notes:

* vectors (norms, biases) always stay in F32;
* a tensor whose row length is not a multiple of the block size (32 or 256) is
  stored in F16;
* nimllm's quantizers are simple versions (min/max per block): to quantize a
  *large* model with the best possible quality, the `llama-quantize` tool (with an
  importance matrix) remains preferable; the files it produces can be read by
  nimllm.

## 12.5 Writing your own GGUF files

`GgufWriter` can store any arrays (vectors of a RAG index, weights of a custom
network…) with metadata:

```nim
# file: write_gguf.nim
import nimllm

let w = newGgufWriter()
w.setKV("general.architecture", gStr("rag-index"))
w.setKV("index.count", gU32(3))
w.setKV("index.sources", gArrStr(["a.txt", "b.txt", "c.txt"]))
w.addTensorF32("vectors", @[4, 3], @[1'f32, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0])  # 3 rows of 4
w.write("index.gguf")

let g = openGguf("index.gguf")
echo g.getStrArray("index.sources"), " ", g.tensors["vectors"].dims
echo g.tensorF32("vectors")
g.close()
```

`dims` follows the ggml convention: `dims[0]` is the length of a row.

## 12.6 Compatibility

| Written by nimllm | Read by |
|---|---|
| models (`saveGguf`, `mergeLora`, `quantizeModel`) | nimllm, llama.cpp, Ollama, LM Studio, GPT4All… |
| LoRA adapters (`saveLora`) | nimllm (`applyLora`) — own format, use `mergeLora` for other tools |

| Read by nimllm | Condition |
|---|---|
| GGUF v2/v3 | llama, mistral, qwen2, qwen3 architectures |
| types F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K…Q6_K | IQ* types are not supported |

**Next:** **13 — Performance and troubleshooting**


---

# 13 — Performance and troubleshooting

## 13.1 Compiler options

| Option | Effect | Recommendation |
|---|---|---|
| `-d:release` | C compiler optimizations | **essential** |
| `--passC:-march=native` | CPU SIMD instructions (AVX2, AVX-512…) | recommended (+20 to 40%); the executable then only runs on equivalent CPUs |
| `-d:danger` | removes all runtime checks | small gain (the compute kernels already disable them); not recommended |
| `--threads:on` | threads (on by default in Nim 2) | do not disable |

Example of a production `config.nims`:

```nim
switch("define", "release")
switch("passC", "-march=native")
switch("opt", "speed")
```

## 13.2 Threads

```nim
# file: threads.nim
import std/os
import nimllm

let model = loadModel(getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
for n in [1, 2, 4, 8]:
  setThreads(n)                           # changes the pool at any time
  let chat = newChat(model, sampling = greedySampling(), maxTokens = 32, lang = "en")
  let r = chat.ask("Count from one to ten.")
  echo n, " thread(s): ", r.tokensPerSecond.int, " tok/s"
```

* By default, nimllm uses all logical cores (`setThreads(0)`).
* Generation is limited by **memory bandwidth**: beyond the number of physical
  cores, the gain is small.
* `loadModel(path, threads = n)` also sets the number of threads.

## 13.3 Orders of magnitude

Measurements on a **2-core** CPU (virtual machine), model with the dimensions of
Llama 3.2 1B in Q4_K_M, `-d:release --passC:-march=native`:

| Operation | nimllm | llama.cpp (same machine) |
|---|---|---|
| Generation | ~7 tokens/s | ~13 tokens/s |
| Prompt processing | ~12 tokens/s | much faster (optimized batched matrix computation) |
| LoRA training | ~4 tokens/s | — |
| Training a 1 M-parameter model | ~3,000 tokens/s | — |

Speed scales roughly with the number of physical cores and inversely with model
size (3B ≈ 2.5 times slower than 1B; Q8_0 ≈ 1.5 times slower than Q4_K_M).

## 13.4 Speed tips

1. **Quantized model**: Q4_K_M is the best compromise; F16/F32 are slow.
2. **Reused context**: keep the same conversation rather than rebuilding the
   prompt; a fixed system prompt is computed only once.
3. **Short prompt**: every prompt token costs computation; summarize long
   histories (chapter 3.5).
4. **Suitable `maxTokens`** and **stop strings** to avoid generating needlessly.
5. **Targeted attachments** (RAG) rather than whole documents.
6. **`nBatch`** (`newContext(m, nBatch = 64)`): larger prompt batches, slightly
   faster, a little more memory.

## 13.5 Common problems

Error messages are in French; their meaning is given in parentheses.

| Symptom | Likely cause | Solution |
|---|---|---|
| Extremely slow | compiled without `-d:release` | add `-d:release` |
| `GgufError: fichier introuvable` (file not found) | wrong path | check the path / `NIMLLM_MODELE` |
| `GgufError: ce n'est pas un fichier GGUF` (not a GGUF file) | `.safetensors` or `.bin` file, HTML download | get the GGUF version of the model |
| `type de tenseur inconnu … IQ*` (unknown tensor type) | "i-quant" quantization | use Q4_K_M, Q5_K_M, Q8_0… |
| `architecture non prise en charge` (unsupported architecture) | Gemma, Phi-3, MoE… | use Llama, Mistral or Qwen |
| Incoherent answers, mixed-up tags | wrong template | force `newChat(m, templ = tplLlama3)` (or the right one); check with `renderPrompt` |
| The model does not stop | *base* model (not Instruct), unknown end of turn | use an "Instruct" model; add `chat.options.stop` |
| Looping repetitions | small model, low temperature | `repeatPenalty` 1.15–1.3, temperature 0.7 |
| `ChatError: message trop long` (message too long) | attachment / question > context | increase `nCtx`, split (chapter 7.4) |
| `ChatError: … JSON valide` (no valid JSON) | model too small or vague instruction | `greedySampling()`, concrete schema, JSON example in the system prompt |
| `ChatError: aucun code SVG` (no SVG code) | the model answered in text | rephrase ("draw…"), larger model |
| `nan` loss during training | learning rate too high | divide `lr` by 3, keep `gradClip = 1.0` |
| Loss does not go down | `lr` too low, data too varied for the model size | increase `lr`, more steps, larger model |
| Out of memory (LoRA) | batches too large | `batchSize = 1`, larger `gradAccum`, smaller `seqLen` |
| "�" characters | cut in the middle of a UTF-8 character | use `onToken`/`text` (already safe) rather than `tokenToPiece` |

## 13.6 Diagnosing

```nim
# file: diagnostic.nim
import std/[os, strutils]
import nimllm

let path = getEnv("NIMLLM_MODELE", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let g = openGguf(path)
echo "Architecture: ", g.getStr("general.architecture")
echo "Tokenizer:    ", g.getStr("tokenizer.ggml.model"), " / ", g.getStr("tokenizer.ggml.pre")
echo "Template:     ", (if g.has("tokenizer.chat_template"): "present" else: "MISSING (force templ)")
g.close()

let m = loadModel(path, verbose = true)
let chat = newChat(m, system = "Test.", lang = "en")
echo "Detected template: ", chat.templ
echo "Prompt sent:\n", renderPrompt(chat.templ, chat.messages & @[Message(role: roleUser, content: "Hello")])
echo "End-of-turn tokens: "
for id in m.tokenizer.eogIds: echo "  ", id, " = ", m.tokenizer.tokens[id]

# Does the model "understand"? The expected tokens should come first:
let ctx = newContext(m, 256)
let logits = ctx.eval(m.tokenizer.encode("The capital of France is", addBos = true))
for (id, p) in topTokens(logits, 3):
  echo "  ", m.tokenizer.tokenToPiece(id).escape, " ", (p*100).formatFloat(ffDecimal, 1), " %"
```

If "Paris" does not come first for a well-known model, the file is probably
corrupted or of a misrecognized architecture: report it with the output of
`g.describe()`.

## 13.7 Checking the installation with the tests

```sh
nim c -r tests/t_complet.nim       # self-contained end-to-end test (~30 s)
nim c -r tests/t_grad.nim          # gradients of every operation
nimble exemples                    # compiles the 20 examples
```

Conformance tests against llama.cpp (require a copy of the llama.cpp repository,
given by the `LLAMA_CPP` variable):

```sh
LLAMA_CPP=~/src/llama.cpp nim c -r tests/t_tok.nim    # tokenizers vs official test suites
```

**Next:** **14 — API reference**


---

# 14 — API reference

`import nimllm` gives access to all the modules below. Signatures use Nim
notation; parameters with `=` are optional.

## chat — conversation

### Types

```nim
type
  Role = enum roleSystem, roleUser, roleAssistant
  Message = object
    role: Role
    content: string
  ChatTemplateKind = enum tplAuto, tplLlama3, tplChatML, tplMistral, tplLlama2, tplGemma, tplPhi3, tplRaw
  OutputFormat = enum ofText, ofMarkdown, ofJson, ofImage, ofAudio, ofFile
  StopReason = enum srEndOfText, srStopString, srMaxTokens, srContextFull, srCallback
  TokenCallback = proc (piece: string): bool {.closure.}
  GenOptions = object
    maxTokens: int; sampling: SamplerParams; stop: seq[string]; onToken: TokenCallback
  Reply = object
    text, raw: string; files: seq[string]; json: JsonNode
    promptTokens, completionTokens: int; stopReason: StopReason
    seconds, tokensPerSecond: float
  Chat = ref object
    model: LlmModel; ctx: LlmContext; system: string; history: seq[Message]
    templ: ChatTemplateKind; options: GenOptions; stripThinking: bool
    lang: string; imageSize: int; jsonRetries: int; voicePitch: float
```

### Functions

| Signature | Description |
|---|---|
| `newChat(m; system = ""; nCtx = 4096; templ = tplAuto; sampling = defaultSampling(); maxTokens = 512; lang = "fr"): Chat` | new conversation |
| `ask(c; prompt; attachments = @[]; format = ofText; outPath = ""; schema = ""; onToken = nil; maxTokens = 0): Reply` | asks a question |
| `askJson(c; prompt; schema = ""): JsonNode` | question with a JSON answer |
| `regenerate(c; onToken = nil): Reply` | new draw of the last answer |
| `continueReply(c; maxTokens = 256; onToken = nil): Reply` | extends the last answer |
| `add(c; role; content)` | adds a message without generating |
| `reset(c)` / `undo(c)` | clears the history / removes the last exchange |
| `setSampling(c; p)` | changes the sampling settings |
| `messages(c): seq[Message]` | system + history |
| `promptTokens(c; addGen = true): seq[int]` | tokenized prompt |
| `saveHistory(c; path)` / `loadHistory(c; path)` / `toJson(c)` | persistence |
| `stats(c): string` | context status |
| `generate(ctx; prompt: seq[int]; opts; smp = nil): Reply` | generation from tokens |
| `complete(ctx; text; opts = defaultOptions()): Reply` | raw completion |
| `defaultOptions(): GenOptions` | 512 tokens, `defaultSampling()` |
| `detectTemplate(tok): ChatTemplateKind` | a model's template |
| `renderPrompt(kind; msgs; addGenerationPrompt = true; bos = "<s>"): string` | prompt text |
| `encodeMessages(tok; kind; msgs; addGenerationPrompt = true): seq[int]` | tokenized prompt (safe tags) |
| `stopStringsFor(kind): seq[string]` | implicit stop strings |
| `extractJson(s): JsonNode` | extracts the first JSON from a text |
| `stripFences(s): string` | removes ``` fences |

## model — inference

| Signature | Description |
|---|---|
| `loadModel(path; threads = 0; verbose = false): LlmModel` | opens a GGUF |
| `describe(m): string` / `paramCount(m): int` / `close(m)` | information / closing |
| `m.cfg: ModelConfig` | `arch, name, dim, hidden, nLayers, nHeads, nKvHeads, headDim, vocab, ctxTrain, ropeBase, ropeDim, ropeNeox, normEps, tiedEmbeddings` |
| `m.tokenizer: Tokenizer` | the model's tokenizer |
| `newContext(m; nCtx = 2048; nBatch = 32): LlmContext` | KV cache |
| `eval(ctx; toks; allLogits = false): seq[float32]` | evaluates tokens |
| `evalPrompt(ctx; toks): seq[float32]` | same, reusing the cached prefix |
| `reset(ctx)` / `truncate(ctx; n)` / `nPast(ctx)` / `ctx.tokens` | cache management |
| `memoryUsage(ctx)` / `tokensPerSecond(ctx)` | statistics |
| `applyLora(m; path)` / `removeLora(m)` | on-the-fly adapters |
| `embed(m; text; nCtx = 512): seq[float32]` | normalized vector |
| `cosineSimilarity(a, b): float` | similarity |
| `perplexity(m; text; nCtx = 512): float` | perplexity |
| `exactMatmul: bool` (global variable) | disables int8 activations |
| `newQMatrix(name; typ; rows, cols; values): QMatrix` / `matmul(w; x; y; nb)` | quantized matrices |

## sampler — sampling

```nim
type SamplerParams = object
  temperature, topP, minP, repeatPenalty, presencePenalty, frequencyPenalty: float
  topK, repeatLastN, seed: int
  logitBias: Table[int, float]
```

| Signature | Description |
|---|---|
| `defaultSampling()` / `greedySampling()` | presets |
| `newSampler(p): Sampler` | sampler |
| `sample(s; logits): int` / `accept(s; tok)` | draw / record |
| `argmax(logits)` / `softmaxInPlace(x)` / `topTokens(logits; n = 5)` | utilities |

## tokenizer — tokens

| Signature | Description |
|---|---|
| `encode(t; text; addBos = false; parseSpecial = true): seq[int]` | text → tokens |
| `decode(t; ids; renderSpecial = false): string` | tokens → text |
| `tokenToPiece(t; id; renderSpecial = false): string` | one token |
| `tokenId(t; text): int` / `isSpecial(t; id)` / `vocabSize(t)` | lookup |
| `t.bosId, t.eosId, t.padId, t.unkId, t.eogIds, t.addBos, t.chatTemplate, t.tokens, t.merges, t.kind, t.pre` | fields |
| `tokenizerFromGguf(g)` / `writeToGguf(t; w)` | GGUF reading / writing |
| `newBpeTokenizer(tokens; merges; types; pre = ptLlama3)` / `rebuild(t)` | construction |
| `preTokenize(text; mode)` / `byteEncode(s)` / `byteDecode(s)` | BPE tools |

## tokentrain — tokenizer training

| Signature | Description |
|---|---|
| `trainBpe(texts; vocabSize = 2048; specials = defaultSpecials; minFreq = 2; pre = ptLlama3; verbose = false): Tokenizer` | byte-level BPE |
| `byteLevelTokenizer(specials = defaultSpecials): Tokenizer` | one token per byte |
| `defaultSpecials` / `chatmlTemplate` | constants |

## attachments — attachments

```nim
type Attachment = object
  name, mime, text, data: string
  kind: AttachmentKind      # akText, akImage, akPdf, akAudio, akBinary
  width, height: int
  image: Image              # pixels for PNG/BMP/PPM
```

| Signature | Description |
|---|---|
| `attach(path): Attachment` | from a file |
| `attachText(name; content)` / `attachData(name; data)` | from memory |
| `toPrompt(a; maxChars = 12000): string` | text inserted in the message |
| `decodePng(data): Image` / `pdfText(data): string` / `describeImage(img)` | decoders |

## image — images

| Signature | Description |
|---|---|
| `rgb(r, g, b; a = 1.0): Color` / `parseColor(s)` | colors |
| `newImage(w, h; bg)` / `getPixel` / `setPixel` | RGB image |
| `writeBmp` / `writePpm` / `save` / `readBmp` / `readPpm` | files |
| `newCanvas(w, h; bg; ss = 3): Canvas` / `finish(c): Image` | anti-aliased drawing |
| `fillRect`, `fillCircle`, `fillPolygon`, `fillPolygons`, `strokePolyline`, `drawLine`, `ellipsePoints` | primitives |
| `renderSvg(svg; width = 0; height = 0; bg): Image` | SVG rendering |
| `extractSvg(text)` / `svgSize(svg)` / `parsePath(d)` | SVG tools |

## audio — sound

| Signature | Description |
|---|---|
| `speak(text; lang = "fr"; pitch = 115.0; speed = 1.0; sampleRate = 16000): Audio` | speech synthesis |
| `renderMelody(notation; bpm = 120.0; sampleRate = 22050): Audio` | melody |
| `writeWav(a; path)` / `readWav(path)` / `readWavInfo(path)` | WAV files |
| `concat(a; b)` / `silence(a; seconds)` / `normalize(a)` / `duration(a)` | editing |
| `textToPhonemes(text; lang)` / `numberToFrench(n)` / `numberToEnglish(n)` / `noteFrequency(note)` | tools |

## autograd — differentiable tensors

| Signature | Description |
|---|---|
| `newTensor(shape; requiresGrad = false)`, `fromSeq(data; shape)`, `full(shape; v)`, `randn(shape; std; rng)`, `param(shape; std; rng; name)`, `scalar(v)` | creation |
| `t.data`, `t.grad`, `t.shape`, `rows`, `cols`, `numel`, `item`, `reshape`, `detach` | access |
| `+`, `-`, `*`, `scale`, `sum`, `mean`, `linear(x, w, b = nil)`, `matmul`, `concatCols` | operations |
| `relu`, `silu`, `gelu`, `sigmoid`, `tanhT`, `softmax`, `rmsnorm`, `dropout` | functions |
| `embedding`, `rope`, `causalAttention`, `crossEntropy`, `mseLoss` | LLM building blocks |
| `backward(loss)` / `zeroGrad(t)` / `noGrad: …` | gradients |
| `needsGrad(…)` / `makeNode(res; parents; backward)` / `ensureGrad(t)` | custom operations |
| `gradCheck(f; t; eps = 1e-2; samples = 10): float` | checking |

## nn — trainable models

| Signature | Description |
|---|---|
| `newModelConfig(vocab; dim = 256; layers = 4; heads = 4; kvHeads = 0; hidden = 0; ctx = 256; ropeBase = 10000.0; tied = true; name): ModelConfig` | architecture |
| `newTransformer(cfg; tok; seed = 42): Transformer` | new model |
| `loadForTraining(path; mode = tmLora; lora = defaultLora(); seed = 42; threads = 0): Transformer` | from a GGUF |
| `defaultLora(): LoraConfig` (`rank`, `alpha`, `targets`) | LoRA settings |
| `parameters(t)` / `parameterCount(t)` | parameters |
| `forward(t; ids; B, T)` / `hidden(t; ids; B, T)` / `loss(t; inputs; targets; B, T)` | computation |
| `generateText(t; prompt; maxTokens = 50; temperature = 0.8)` | monitoring generation |
| `saveGguf(t; path; wtype = gtF32)` / `saveLora(t; path)` | export |
| `mergeLora(base; adapter; out; outType = gtQ8_0)` / `quantizeModel(in; out; wtype; keepOutput = true)` | files |
| `qlinear(x; q)` / `qembedding(q; ids)` / `Linear.forward(x)` / `addLora(l; rank; alpha; rng)` | layers |

## train — training

| Signature | Description |
|---|---|
| `newAdamW(params; lr = 3e-4; beta1 = 0.9; beta2 = 0.95; eps = 1e-8; weightDecay = 0.1)` / `newSGD(params; lr; momentum; weightDecay)` | optimizers |
| `zeroGrad(o)` / `update(o)` / `saveState(o; path)` / `loadState(o; path)` / `o.lr`, `o.step` | optimizer |
| `clipGradNorm(params; maxNorm)` / `gradNorm(params)` / `cosineLr(step, warmup, total, maxLr, minLr)` | tools |
| `newTextDataset(tok; text)` / `newTextDatasetFromFiles(tok; paths)` | text data |
| `newChatDataset(tok; templ; conversations)` / `loadChatJsonl(path; tok; templ; system = "")` / `addChatExample` | dialogues |
| `split(d; valFraction = 0.1)` / `getBatch(d; B, T): Batch` / `len(d)` | handling |
| `defaultTrainConfig(): TrainConfig` | `steps, batchSize, seqLen, gradAccum, lr, minLr, warmup, weightDecay, gradClip, evalEvery, evalBatches, logEvery, saveEvery, savePath, sampleEvery, samplePrompt, seed` |
| `train(t; data; cfg; valData = nil; opt = nil; onLog = nil): Optimizer` | full loop |
| `evaluate(t; d; batches = 4; B = 8; T = 64): float` | average loss |
| `saveCheckpoint(t; opt; prefix)` / `printLog(l)` | saving / logging |

## gguf and quant — files and formats

| Signature | Description |
|---|---|
| `openGguf(path): GgufFile` / `close(g)` / `describe(g)` | reading |
| `g.kv`, `g.tensors`, `has`, `getInt`, `getFloat`, `getStr`, `getBool`, `getStrArray`, `getIntArray`, `getFloatArray` | metadata |
| `tensorF32(g; name)`, `GgufTensorInfo.dims/typ/data/numElements/byteSize` | tensors |
| `newGgufWriter()`, `setKV`, `addTensor`, `addTensorF32(name; dims; values; typ)`, `addTensorRaw`, `copyMetadata`, `write` | writing |
| `gStr`, `gU32`, `gI32`, `gU64`, `gF32`, `gBool`, `gArrStr`, `gArrI32`, `gArrF32` | values |
| `GgmlType` (`gtF32`, `gtF16`, `gtBF16`, `gtQ8_0`, `gtQ4_0`… `gtQ6_K`), `parseGgmlType("q4_k")` | types |
| `quantize(t; data)`, `quantizeRow`, `dequantRow`, `blockSize`, `typeSize`, `rowBytes`, `bitsPerWeight` | conversions |
| `floatToHalf` / `halfToFloat` / `floatToBf16` / `bf16ToFloat` | half precision |

## parallel — threads

| Signature | Description |
|---|---|
| `setThreads(n = 0)` / `numThreads()` / `shutdownPool()` | pool |
| `parallelFor(n; fn: TaskFn; ctx: pointer; minChunk = 1)` | parallel loop (`fn(ctx, first, last, worker)`) |

## inflate — decompression

| Signature | Description |
|---|---|
| `zlibDecompress(data)` / `inflateRaw(data)` | zlib / DEFLATE stream |

<br/><br/>

> © 2026 Jean-Marc Quéré, sonaliwan.fr - Linguistics & Technologies<br/>
All rights reserved<br/>
SIRET: 123 456 789 00012<br/>
License: CC BY-NC-SA 4.0<br/>
