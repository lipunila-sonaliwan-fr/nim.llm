# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 08 — Recevoir la réponse sous forme d'image.
#
#   nim c -r examples/ex08_image.nim modele.gguf
#
# Un LLM textuel ne produit pas de pixels : on lui fait écrire du **SVG**
# (dessin vectoriel en texte), que nimllm rastérise en BMP. On obtient deux
# fichiers : `.svg` (vectoriel, ouvrable dans un navigateur) et `.bmp`.

import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)
let conv = newChat(modele, maxTokens = 1500)
conv.imageSize = 512             # largeur de l'image produite (pixels).

# 1. Demande d'image.
let r = conv.ask("Dessine une maison avec un toit rouge, un soleil et un arbre.",
                 format = ofImage, outPath = "maison.svg")
echo "Fichiers créés : ", r.files          # @["maison.svg", "maison.bmp"]

# 2. Un graphique à partir de données.
conv.reset()
let r2 = conv.ask("Fais un diagramme en barres (SVG) des ventes : janvier 120, " &
                  "février 95, mars 150. Ajoute un axe horizontal.",
                  format = ofImage, outPath = "ventes.svg")
echo "Graphique : ", r2.files

# 3. Rendu SVG sans LLM : le moteur est utilisable directement.
let svg = """<svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">
  <rect width="200" height="120" fill="#f5f0e6"/>
  <path d="M10 100 C 60 10, 140 10, 190 100" fill="none" stroke="crimson" stroke-width="4"/>
  <circle cx="100" cy="35" r="12" fill="orange" stroke="black"/>
  <g transform="translate(20,20) rotate(15)"><rect width="30" height="20" fill="teal" rx="4"/></g>
</svg>"""
renderSvg(svg, 400).writeBmp("courbe.bmp")

# 4. Dessiner soi-même avec Canvas.
let c = newCanvas(300, 200, rgb(255, 255, 255))
c.fillRect(0, 150, 300, 50, rgb(80, 160, 60))                     # herbe.
c.fillCircle(240, 50, 30, rgb(255, 200, 0))                       # soleil.
c.fillPolygon(@[(60.0, 150.0), (110.0, 90.0), (160.0, 150.0)], rgb(180, 40, 40))
c.drawLine(10, 10, 290, 10, 3, rgb(0, 0, 0, 0.5))                 # trait semi-transparent.
let image = c.finish()
image.writeBmp("dessin.bmp")
image.writePpm("dessin.ppm")
echo "Pixel (240, 50) = ", image.getPixel(240, 50)
