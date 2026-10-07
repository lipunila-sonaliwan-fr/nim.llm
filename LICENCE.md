# Licence de nimllm

## 1. Licence principale : CC BY-NC-SA 4.0

Sauf mention contraire à la section 3, **l'ensemble du projet nimllm** est
distribué sous la licence **Creative Commons Attribution – Pas d'utilisation
commerciale – Partage dans les mêmes conditions 4.0 International
(CC BY-NC-SA 4.0)**. Cela couvre le code source, les exemples, les tests, les
données d'exemple et la documentation.

* Titulaire des droits : **sonaliwan.fr**
* Résumé : <https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr>
* Texte juridique : <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.fr>
* Identifiant SPDX : `CC-BY-NC-SA-4.0`

Le résumé ci-dessous n'a pas de valeur juridique : seul le texte juridique fait foi.
Il vous autorise à :

* **partager** : copier et redistribuer le contenu, sur tout support et en tout format ;
* **adapter** : modifier, transformer et créer à partir du contenu.

Ces autorisations sont soumises aux conditions suivantes :

* **Attribution (BY)** : vous devez créditer « nimllm – sonaliwan.fr », fournir un
  lien vers la licence et indiquer si des modifications ont été apportées ;
* **Pas d'utilisation commerciale (NC)** : vous ne pouvez pas utiliser le contenu
  à des fins commerciales ;
* **Partage dans les mêmes conditions (SA)** : si vous modifiez le contenu ou
  créez à partir de lui, vous devez diffuser votre contribution sous la même
  licence ;
* **Pas de restrictions supplémentaires** : vous ne pouvez pas ajouter de
  conditions juridiques ou de mesures techniques qui empêcheraient autrui de
  faire ce que la licence autorise.

Mention de crédit suggérée :

> nimllm – © sonaliwan.fr – licence CC BY-NC-SA 4.0
> (https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 2. Licence commerciale

Pour toute utilisation commerciale de la partie de nimllm placée sous
CC BY-NC-SA 4.0, une **licence commerciale** peut être acquise auprès de :

| | |
|---|---|
| Société | **sonaliwan.fr** |
| SIRET | **130333198000013** |
| Contact | **metalab (at) sonaliwan.fr** |

La licence commerciale porte uniquement sur les parties dont sonaliwan.fr est
titulaire. Les parties listées à la section 3 restent régies par leurs propres
licences, qui autorisent toutes l'usage commercial sous réserve d'en respecter
les conditions (conservation des mentions de copyright et des textes de licence).

## 3. Parties soumises à d'autres licences

Les éléments suivants ne sont **pas** couverts par la licence CC BY-NC-SA 4.0
(ou ne le sont que pour les apports de nimllm). Ils restent sous la licence de
leur détenteur. Les textes complets de ces licences se trouvent dans le dossier
[`LICENSES/`](LICENSES/).

| # | Élément | Fichiers de nimllm concernés | Détenteur | Licence | Texte |
|---|---|---|---|---|---|
| 3.1 | Portions dérivées de **ggml / llama.cpp** : disposition binaire des blocs quantifiés (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K à Q8_K), algorithmes de déquantification et d'empaquetage des échelles, produits scalaires par blocs, format de fichier GGUF (lecture/écriture), conventions de nommage des tenseurs et des métadonnées, algorithme du tokeniseur SentencePiece, algorithme de fusion BPE, règles de pré-découpage (expressions « llama3 », « qwen2 », « gpt2 ») et correspondances des noms de pré-tokeniseurs | `src/nimllm/quant.nim`, `src/nimllm/gguf.nim`, `src/nimllm/tokenizer.nim` ; conventions reprises dans `src/nimllm/model.nim` et `src/nimllm/nn.nim` | The ggml authors (© 2023-2026) | MIT | [`LICENSES/MIT-ggml.txt`](LICENSES/MIT-ggml.txt) |
| 3.2 | Correspondance octets ↔ caractères Unicode du BPE niveau octet (fonction `bytes_to_unicode` de GPT-2) | `src/nimllm/tokenizer.nim` (`initByteMap`, `byteToUni`, `uniToByte`) | OpenAI (© 2019) | Modified MIT License | [`LICENSES/MIT-OpenAI-GPT2.txt`](LICENSES/MIT-OpenAI-GPT2.txt) |
| 3.3 | Tables de catégories Unicode (lettres L*, nombres N*) générées à partir de l'Unicode Character Database, version 15.1.0 | `src/nimllm/unicode_tables.nim` | Unicode, Inc. (© 1991-2024) | Unicode License v3 | [`LICENSES/Unicode-3.0.txt`](LICENSES/Unicode-3.0.txt) |
| 3.4 | Décodage DEFLATE par codes de Huffman canoniques, structure inspirée de `puff.c` (distribution zlib) | `src/nimllm/inflate.nim` | Mark Adler (© 2002-2013) | zlib | [`LICENSES/Zlib-puff.txt`](LICENSES/Zlib-puff.txt) |

Précisions :

* Pour les fichiers des lignes 3.1 à 3.4, la licence tierce s'applique aux
  portions dérivées de l'œuvre d'origine. Les apports propres à nimllm dans ces
  mêmes fichiers (adaptation en Nim, commentaires, fonctions ajoutées) sont
  placés sous CC BY-NC-SA 4.0. Toute redistribution de ces fichiers doit
  conserver les mentions de copyright et les textes de licence tiers.
* Les versions modifiées de l'œuvre d'origine sont signalées ici comme telles,
  conformément à la licence zlib (3.4).

## 4. Éléments externes non distribués

Ces éléments ne font pas partie de nimllm et ne sont pas redistribués avec lui.

* **Modèles de langage** (Llama 3.2, Mistral, Qwen…) : chaque modèle reste
  soumis à la licence de son éditeur, par exemple la *Llama 3.2 Community
  License* de Meta. La licence de nimllm ne s'étend pas aux modèles chargés avec
  la bibliothèque.
* **Gabarits de dialogue et noms de tokens spéciaux** (`<|start_header_id|>`,
  `<|im_start|>`, `[INST]`…) : conventions d'interopérabilité définies par les
  éditeurs des modèles, reproduites uniquement pour assurer la compatibilité.
* **Fichiers de test de llama.cpp** (vocabulaires `ggml-vocab-*.gguf`) :
  utilisés par `tests/t_tok.nim` depuis une copie locale de llama.cpp
  (variable `LLAMA_CPP`), et non inclus dans nimllm.
* **Fichiers produits par vos utilisateurs avec nimllm** (modèles entraînés,
  adaptateurs LoRA, textes, images, sons générés) : leur statut dépend des
  données et des modèles utilisés pour les produire.

## 5. Absence de garantie

nimllm est fourni « en l'état », sans garantie d'aucune sorte, conformément à la
section 5 de la licence CC BY-NC-SA 4.0 et aux clauses équivalentes des licences
tierces citées.
