# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 03 — Définir un contexte (prompt système), dialoguer sur plusieurs
# tours, sauvegarder puis recharger la conversation.
#
#   nim c -r examples/ex03_conversation.nim modele.gguf

import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)

# Le prompt système fixe le rôle, le ton et les règles du modèle.
let conv = newChat(modele,
  system = """Tu es Marcel, sommelier français à la retraite.
Tu réponds en deux phrases maximum, avec humour, et tu tutoies ton interlocuteur.""",
  nCtx = 4096,                   # mémoire de la conversation (en tokens).
  maxTokens = 200)               # longueur maximale de chaque réponse.

proc tour(question: string) =
  echo "Vous    : ", question
  echo "Marcel  : ", conv.ask(question).text
  echo ""

tour("Bonjour ! Quel vin avec une raclette ?")
tour("Et si je n'aime pas le vin blanc ?")                        # le modèle se souvient du tour précédent.
tour("Rappelle-moi ce que je t'ai demandé en premier.")

echo "État : ", conv.stats

# Sauvegarde / reprise.
conv.saveHistory("conversation_marcel.json")

let reprise = newChat(modele)
reprise.loadHistory("conversation_marcel.json")                   # restaure système + messages.
echo "Messages rechargés : ", reprise.history.len
echo "Marcel (repris) : ", reprise.ask("Et pour le dessert ?").text

# Annuler / régénérer.
reprise.undo()                                                    # retire le dernier échange.
echo "Après undo : ", reprise.history.len, " messages"

discard reprise.ask("Un vin pour accompagner des huîtres ?")
echo "Autre proposition : ", reprise.regenerate().text            # nouveau tirage.

# Changer de rôle en cours de route.
reprise.system = "Tu es un professeur de chimie rigoureux."
reprise.reset()                                                   # oublie l'historique, garde le système.
echo reprise.ask("Pourquoi le vin vieillit-il ?").text

# Exemples « few-shot » : on montre au modèle le format attendu.
let classeur = newChat(modele, system = "Classe le sentiment : POSITIF, NEGATIF ou NEUTRE.",
                       sampling = greedySampling(), maxTokens = 5)
classeur.add(roleUser, "Ce restaurant est formidable !")
classeur.add(roleAssistant, "POSITIF")
classeur.add(roleUser, "Le service était lent et froid.")
classeur.add(roleAssistant, "NEGATIF")
echo "Sentiment : ", classeur.ask("Le plat était correct, sans plus.").text
