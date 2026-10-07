# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 16 — Entraîner son propre tokeniseur BPE (compatible llama.cpp).
#
#   nim c -r examples/ex16_tokeniseur.nim [corpus.txt]

import std/[os, strutils]
import nimllm

let corpus = if paramCount() >= 1: readFile(paramStr(1))
             else: """Le petit prince demanda : « S'il vous plaît... dessine-moi un mouton ! »
Les grandes personnes ne comprennent jamais rien toutes seules, et c'est fatigant,
pour les enfants, de toujours et toujours leur donner des explications.
On ne voit bien qu'avec le cœur. L'essentiel est invisible pour les yeux.""".repeat(20)

# Apprentissage : 256 octets de base + fusions + tokens spéciaux.
let tok = trainBpe([corpus], vocabSize = 600,
                   specials = @["<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"],
                   minFreq = 2)
echo "Vocabulaire : ", tok.vocabSize, " tokens, ", tok.merges.len, " fusions"
echo "Spéciaux : bos=", tok.bosId, " eos=", tok.eosId, " pad=", tok.padId
echo "Fin de tour (eog) : ", tok.eogIds

let phrase = "L'essentiel est invisible pour les yeux."
let ids = tok.encode(phrase)
echo "\n", phrase
echo "→ ", ids.len, " tokens : ", ids
var morceaux: seq[string]
for id in ids: morceaux.add tok.tokenToPiece(id)
echo "→ découpage : ", morceaux.join(" | ")
assert tok.decode(ids) == phrase                                  # l'aller-retour est exact.

# Le niveau octet garantit que TOUT texte est encodable (même inconnu) :
let exotique = "Emoji 🦙, grec αβγ, chinois 你好"
assert tok.decode(tok.encode(exotique)) == exotique
echo "Texte inconnu : ", tok.encode(exotique).len, " tokens (repli sur les octets)"

# Sauvegarde dans un GGUF (sans poids) : réutilisable plus tard.
let w = newGgufWriter()
w.setKV("general.architecture", gStr("llama"))
tok.writeToGguf(w)
w.write("mon_tokeniseur.gguf")
let relu = tokenizerFromGguf(openGguf("mon_tokeniseur.gguf"))
assert relu.encode(phrase) == ids
echo "Tokeniseur rechargé depuis mon_tokeniseur.gguf : identique."

# Le pré-découpage (même règle que Llama 3) avant le BPE :
echo preTokenize("Il a 12345 euros, n'est-ce pas ?", ptLlama3)
