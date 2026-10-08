# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 06 — Poser une question accompagnée de pièces jointes.
#
#   nim c -r examples/ex06_pieces_jointes.nim modele.gguf
#
# Formats pris en charge : texte/code/CSV/JSON/Markdown/HTML (contenu),
# PDF (texte extrait), PNG/BMP/PPM (couleurs + aperçu), JPEG/GIF/WebP
# (dimensions), WAV (durée et format). Rappel : Llama 3.2 1B/3B est un
# modèle *textuel* : une image lui est décrite, il ne la « voit » pas.

import std/[os, strutils]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)
let conv = newChat(modele, nCtx = 8192, maxTokens = 400)

# 1. Un fichier texte du disque.
writeFile("ventes.csv", """mois,produit,quantite,prix
janvier,stylo,120,1.5
janvier,cahier,40,3.2
fevrier,stylo,95,1.5
fevrier,cahier,60,3.2
mars,stylo,150,1.4
""")
let r1 = conv.ask("Quel produit rapporte le plus ? Donne le chiffre d'affaires par produit.",
                  attachments = @[attach("ventes.csv")])
echo "CSV → ", r1.text, "\n"

# 2. Plusieurs pièces jointes, dont une construite en mémoire.
conv.reset()
let note = attachText("note.md", "# Réunion du 3 mars\n- Budget validé : 12 000 €\n- Lancement : avril")
let code = attachText("calcul.nim", "proc moyenne(x: seq[float]): float =\n  for v in x: result += v\n  result / x.len.float")
let r2 = conv.ask("Résume la note, puis signale le bug du code.", attachments = @[note, code])
echo "Note + code → ", r2.text, "\n"

# 3. Une image (ici générée pour l'exemple).
conv.reset()
let dessin = renderSvg("""<svg viewBox="0 0 100 100"><rect width="100" height="100" fill="skyblue"/>
  <circle cx="75" cy="25" r="15" fill="gold"/><rect y="70" width="100" height="30" fill="green"/></svg>""", 200, 200)
dessin.writeBmp("paysage.bmp")
let img = attach("paysage.bmp")
echo "Ce que le modèle reçoit pour l'image :\n", img.toPrompt(), "\n"
echo "Image → ", conv.ask("Décris cette image et devine ce qu'elle représente.",
                          attachments = @[img]).text, "\n"

# 4. Un PDF.
# let pdf = attach("contrat.pdf")       # texte extrait automatiquement
# echo conv.ask("Quelles sont les dates clés de ce contrat ?", attachments = @[pdf]).text

# 5. Contrôler la taille insérée dans le prompt.
# Les gros fichiers sont tronqués à 12 000 caractères par défaut ; pour un
# contrôle fin, construisez vous-même le texte :
let gros = attachText("journal.log", "ligne de journal\n".repeat(5000))
echo "Taille insérée : ", gros.toPrompt(maxChars = 2000).len, " octets"
