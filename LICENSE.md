# nimllm License

**English** · [Français](LICENCE.md)

## 1. Main license: CC BY-NC-SA 4.0

Unless otherwise stated in section 3, **the entire nimllm project** is
distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0)** license. This covers the source code, the
examples, the tests, the sample data and the documentation.

* Rights holder: **sonaliwan.fr**
* Summary: <https://creativecommons.org/licenses/by-nc-sa/4.0/>
* Legal code: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>
* SPDX identifier: `CC-BY-NC-SA-4.0`

The summary below has no legal value: only the legal code is authoritative.
It allows you to:

* **share**: copy and redistribute the material in any medium or format;
* **adapt**: remix, transform and build upon the material.

These permissions are subject to the following terms:

* **Attribution (BY)**: you must credit "nimllm – sonaliwan.fr", provide a link
  to the license and indicate if changes were made;
* **NonCommercial (NC)**: you may not use the material for commercial purposes;
* **ShareAlike (SA)**: if you remix, transform or build upon the material, you
  must distribute your contributions under the same license;
* **No additional restrictions**: you may not apply legal terms or technological
  measures that prevent others from doing anything the license permits.

Suggested credit line:

> nimllm – © sonaliwan.fr – CC BY-NC-SA 4.0 license
> (https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 2. Commercial license

For any commercial use of the part of nimllm licensed under CC BY-NC-SA 4.0, a
**commercial license** can be purchased from:

| | |
|---|---|
| Company | **sonaliwan.fr** |
| SIRET | **1303331980000130** |
| Contact | **metalab@sonaliwan.fr** |

The commercial license only covers the parts for which sonaliwan.fr holds the
rights. The parts listed in section 3 remain governed by their own licenses, all
of which permit commercial use provided their terms are met (keeping the
copyright notices and the license texts).

## 3. Parts under other licenses

The following items are **not** covered by the CC BY-NC-SA 4.0 license (or are
covered only for nimllm's own contributions). They remain under their holder's
license. The full texts of these licenses are in the [`LICENSES/`](LICENCES/)
folder.

| # | Item | nimllm files concerned | Holder | License | Text |
|---|---|---|---|---|---|
| 3.1 | Portions derived from **ggml / llama.cpp**: binary layout of quantized blocks (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q8_K), dequantization and scale-packing algorithms, block dot products, GGUF file format (reading/writing), tensor and metadata naming conventions, SentencePiece tokenizer algorithm, BPE merge algorithm, pre-tokenization rules ("llama3", "qwen2", "gpt2" expressions) and pre-tokenizer name mappings | `src/nimllm/quant.nim`, `src/nimllm/gguf.nim`, `src/nimllm/tokenizer.nim`; conventions reused in `src/nimllm/model.nim` and `src/nimllm/nn.nim` | The ggml authors (© 2023-2026) | MIT | [`LICENSES/MIT-ggml.txt`](LICENSES/MIT-ggml.txt) |
| 3.2 | Byte ↔ Unicode character mapping of byte-level BPE (GPT-2's `bytes_to_unicode` function) | `src/nimllm/tokenizer.nim` (`initByteMap`, `byteToUni`, `uniToByte`) | OpenAI (© 2019) | Modified MIT License | [`LICENSES/MIT-OpenAI-GPT2.txt`](LICENSES/MIT-OpenAI-GPT2.txt) |
| 3.3 | Unicode category tables (letters L*, numbers N*) generated from the Unicode Character Database, version 15.1.0 | `src/nimllm/unicode_tables.nim` | Unicode, Inc. (© 1991-2024) | Unicode License v3 | [`LICENSES/Unicode-3.0.txt`](LICENSES/Unicode-3.0.txt) |
| 3.4 | DEFLATE decoding with canonical Huffman codes, structure inspired by `puff.c` (zlib distribution) | `src/nimllm/inflate.nim` | Mark Adler (© 2002-2013) | zlib | [`LICENSES/Zlib-puff.txt`](LICENSES/Zlib-puff.txt) |

Clarifications:

* For the files in rows 3.1 to 3.4, the third-party license applies to the
  portions derived from the original work. nimllm's own contributions in those
  same files (Nim adaptation, comments, added functions) are licensed under
  CC BY-NC-SA 4.0. Any redistribution of these files must keep the third-party
  copyright notices and license texts.
* Modified versions of the original work are marked as such here, as required by
  the zlib license (3.4).

## 4. External items not distributed

These items are not part of nimllm and are not redistributed with it.

* **Language models** (Llama 3.2, Mistral, Qwen…): each model remains subject to
  its publisher's license, for example Meta's *Llama 3.2 Community License*.
  nimllm's license does not extend to the models loaded with the library.
* **Chat templates and special token names** (`<|start_header_id|>`,
  `<|im_start|>`, `[INST]`…): interoperability conventions defined by the model
  publishers, reproduced solely to ensure compatibility.
* **llama.cpp test files** (`ggml-vocab-*.gguf` vocabularies): used by
  `tests/t_tok.nim` from a local copy of llama.cpp (`LLAMA_CPP` variable), and
  not included in nimllm.
* **Files produced by users with nimllm** (trained models, LoRA adapters,
  generated text, images and sounds): their status depends on the data and the
  models used to produce them.

## 5. No warranty

nimllm is provided "as is", without warranty of any kind, in accordance with
section 5 of the CC BY-NC-SA 4.0 license and the equivalent clauses of the
third-party licenses mentioned above.
