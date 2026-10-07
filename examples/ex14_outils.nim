# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 14 — Donner des « outils » au modèle (appel de fonctions).
#
#   nim c -r examples/ex14_outils.nim modele.gguf
#
# Le modèle décide, en JSON, s'il faut appeler une fonction Nim ; le
# programme l'exécute et renvoie le résultat au modèle, qui formule la
# réponse finale. C'est le principe des « agents ».

import std/[os, json, strutils, times, math]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)

# Les outils disponibles.
proc outilHeure(args: JsonNode): string = now().format("HH:mm")

proc outilCalcul(args: JsonNode): string =
  ## Calculatrice minimaliste : a <op> b
  let a = args{"a"}.getFloat
  let b = args{"b"}.getFloat
  case args{"operation"}.getStr
  of "+": $(a + b)
  of "-": $(a - b)
  of "*": $(a * b)
  of "/": (if b == 0: "division par zéro" else: $(a / b))
  of "^": $pow(a, b)
  else: "opération inconnue"

proc outilMeteo(args: JsonNode): string =
  # Ici on simule ; en réalité, interrogez votre propre source de données.
  "Ensoleillé, 22 °C à " & args{"ville"}.getStr("?")

const description = """Tu peux utiliser ces outils :
- heure() : heure actuelle
- calcul(a, b, operation) : opération parmi + - * / ^
- meteo(ville) : météo d'une ville
Si un outil est utile, réponds UNIQUEMENT avec un JSON :
{"outil": "nom", "arguments": {...}}
Sinon, réponds UNIQUEMENT avec : {"outil": "aucun", "reponse": "ta réponse"}"""

var p = defaultSampling()
p.temperature = 0.1
let agent = newChat(modele, system = description, sampling = p, maxTokens = 200)

proc demander(question: string): string =
  agent.reset()
  let decision = agent.ask(question, format = ofJson)
  let outil = decision.json{"outil"}.getStr("aucun")
  echo "  [décision] ", decision.json
  if outil == "aucun":
    return decision.json{"reponse"}.getStr(decision.text)
  let args = decision.json{"arguments"}
  let resultat = case outil
    of "heure": outilHeure(args)
    of "calcul": outilCalcul(args)
    of "meteo": outilMeteo(args)
    else: "outil inconnu"
  echo "  [résultat de ", outil, "] ", resultat
  # On renvoie le résultat ; le modèle rédige la réponse finale en texte libre.
  agent.add(roleUser, "Résultat de l'outil " & outil & " : " & resultat &
            "\nRéponds maintenant à la question initiale en une phrase, sans JSON.")
  let final = agent.ctx.generate(agent.promptTokens(), agent.options)
  final.text.strip

for q in ["Combien font 1234 multiplié par 56 ?", "Quel temps fait-il à Nantes ?",
          "Quelle heure est-il ?", "Qui a écrit Les Misérables ?"]:
  echo "Q : ", q
  try:
    echo "R : ", demander(q), "\n"
  except CatchableError as e:
    echo "Erreur : ", e.msg, "\n"
