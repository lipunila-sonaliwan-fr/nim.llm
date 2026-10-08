# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 07 — Obtenir une réponse structurée en JSON (extraction d'informations).
#
#   nim c -r examples/ex07_json.nim modele.gguf

import std/[os, json]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)

# Une température basse rend la sortie plus régulière.
var p = defaultSampling()
p.temperature = 0.2
let conv = newChat(modele, sampling = p, maxTokens = 300)

# 1. Forme courte : askJson retourne directement un JsonNode.
let villes = conv.askJson("Donne trois grandes villes d'Italie avec leur région.",
  schema = """{"villes": [{"nom": "texte", "region": "texte"}]}""")
# Accès prudent : {} retourne nil si la clé manque, getStr("") ne lève jamais.
let liste = villes{"villes"}
if liste != nil and liste.kind == JArray:
  for v in liste:
    echo "- ", v{"nom"}.getStr("?"), " (", v{"region"}.getStr("?"), ")"

# 2. Forme complète : ask(..., format = ofJson).
conv.reset()
let courriel = """Bonjour, je m'appelle Julie Martin, j'ai commandé le 12 mars
une bouilloire (réf. BX-220) qui ne chauffe plus. Pouvez-vous me rembourser ?
Mon téléphone : 06 12 34 56 78."""
let r = conv.ask("Extrais les informations de ce message client :\n" & courriel,
  format = ofJson,
  schema = """{"nom": str, "date_commande": str, "produit": str, "reference": str,
               "demande": "remboursement|échange|information", "telephone": str}""")
echo r.json.pretty
echo "Référence : ", r.json{"reference"}.getStr("?")

# 3. Conversion vers un objet Nim typé.
type Fiche = object
  nom: string
  produit: string
  reference: string
try:
  let f = r.json.to(Fiche)
  echo "Objet Nim : ", f
except CatchableError as e:
  echo "Champ manquant : ", e.msg

# 4. Si le modèle échoue.
# ask réessaie automatiquement (conv.jsonRetries = 2 par défaut, avec une
# température divisée par deux) puis lève ChatError.
conv.jsonRetries = 3
try:
  discard conv.askJson("Liste les jours de la semaine.", schema = """{"jours": [str]}""")
except ChatError as e:
  echo "Échec : ", e.msg

# 5. Extraire du JSON d'un texte quelconque.
echo extractJson("Voici le résultat : ```json\n{\"ok\": true}\n``` Bonne journée !")
