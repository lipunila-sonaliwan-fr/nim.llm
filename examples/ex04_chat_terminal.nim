# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 04 — Un assistant complet dans le terminal.
#
#   nim c -r examples/ex04_chat_terminal.nim modele.gguf
#
# Commandes disponibles pendant la discussion :
#   /system <texte>     change le prompt système (et efface l'historique)
#   /joindre <fichier>  joint un fichier au prochain message
#   /format <f>         texte | markdown | json | image | audio | fichier
#   /temp <valeur>      change la température (0 = déterministe)
#   /sauver <f.json>    enregistre la conversation
#   /charger <f.json>   recharge une conversation
#   /annuler            retire le dernier échange
#   /refaire            régénère la dernière réponse
#   /reset              nouvelle conversation
#   /stats              état du contexte
#   /quitter

import std/[os, strutils]
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODEL", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

echo "Chargement de ", chemin, " ..."
let modele = loadModel(chemin, verbose = true)
let conv = newChat(modele, system = "Tu es un assistant serviable qui répond en français.",
                   nCtx = 4096, maxTokens = 1024)

var joints: seq[Attachment]
var format = ofText
var compteur = 0

proc afficher(s: string): bool =
  stdout.write s
  stdout.flushFile()
  true

echo "Prêt. Tapez /quitter pour sortir.\n"
while true:
  stdout.write "\e[1mVous>\e[0m "
  var ligne: string
  if not stdin.readLine(ligne): break
  ligne = ligne.strip
  if ligne.len == 0: continue
  if ligne.startsWith("/"):
    let parts = ligne.split(maxsplit = 1)
    let arg = if parts.len > 1: parts[1] else: ""
    try:
      case parts[0]
      of "/quitter", "/q": break
      of "/system":
        conv.system = arg; conv.reset(); echo "Prompt système modifié."
      of "/joindre":
        joints.add attach(arg); echo "Joint : ", arg, " (", joints[^1].kind, ")"
      of "/format":
        format = parseEnum[OutputFormat](arg); echo "Format : ", format
      of "/temp":
        var p = conv.options.sampling
        p.temperature = parseFloat(arg)
        conv.setSampling(p); echo "Température : ", p.temperature
      of "/sauver": conv.saveHistory(arg); echo "Enregistré."
      of "/charger": conv.loadHistory(arg); echo "Chargé : ", conv.history.len, " messages."
      of "/annuler": conv.undo(); echo "Dernier échange retiré."
      of "/refaire":
        stdout.write "Assistant> "
        discard conv.regenerate(onToken = afficher); echo ""
      of "/reset": conv.reset(); echo "Nouvelle conversation."
      of "/stats": echo conv.stats
      else: echo "Commande inconnue."
    except CatchableError as e:
      echo "Erreur : ", e.msg
    continue
  # message normal.
  inc compteur
  let sortie = case format
    of ofImage: "image_" & $compteur & ".svg"
    of ofAudio: "reponse_" & $compteur & ".wav"
    of ofFile: "fichier_" & $compteur & ".txt"
    else: ""
  stdout.write "\e[1mAssistant>\e[0m "
  try:
    let enFlux = format in {ofText, ofMarkdown}
    let r = conv.ask(ligne, attachments = joints, format = format, outPath = sortie,
                     onToken = (if enFlux: afficher else: nil))
    if not enFlux: echo r.text
    echo ""
    for f in r.files: echo "  → fichier créé : ", f
  except CatchableError as e:
    echo "\nErreur : ", e.msg
  joints.setLen(0)
