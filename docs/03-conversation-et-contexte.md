# 03 — Conversation et contexte

Objectif : définir le **contexte** (rôle, consignes), dialoguer sur plusieurs tours,
maîtriser la mémoire de la conversation.

## 3.1 Le prompt système

Le prompt système est une consigne permanente placée avant l'historique. C'est
le moyen principal de définir le comportement du modèle : rôle, ton, langue,
format, règles, connaissances de référence.

```nim
# fichier : role.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

let guide = newChat(modele, system = """
Tu es Léa, guide touristique à Lyon.
- Tu réponds en français, en 3 phrases maximum.
- Tu proposes toujours une adresse précise.
- Si on te parle d'une autre ville, tu ramènes poliment la conversation sur Lyon.""")

echo guide.ask("Où manger une spécialité locale ?").text
echo guide.ask("Et à Marseille ?").text
```

Conseils pour un bon prompt système :

* soyez **explicite** (« 3 phrases maximum » plutôt que « sois bref ») ;
* donnez des **exemples** du format attendu ;
* placez les informations de référence (tarifs, horaires…) dans le prompt système
  ou en pièce jointe ;
* pour un petit modèle (1B), préférez des consignes courtes et simples.

Le prompt système peut être changé à tout moment : `conv.system = "..."`.

## 3.2 Dialogue sur plusieurs tours

Chaque appel à `ask` ajoute la question et la réponse à `conv.history`. Le modèle
voit donc toute la conversation :

```nim
# fichier : dialogue.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, system = "Tu es un professeur de mathématiques patient.")

for question in ["Qu'est-ce qu'un nombre premier ?",
                 "Donne-moi les cinq premiers.",
                 "Et le suivant après le dernier que tu as cité ?"]:
  echo "Élève : ", question
  echo "Prof  : ", conv.ask(question).text, "\n"

# Inspection de l'historique
for m in conv.history:
  echo "[", m.role, "] ", m.content[0 ..< min(60, m.content.len)]
echo conv.stats
```

### Efficacité : le cache KV

Le modèle mémorise les calculs déjà faits sur le début de la conversation (le
*cache clé/valeur*). À chaque tour, nimllm compare le nouveau prompt au contenu
du cache et ne calcule que la partie nouvelle : les longues conversations restent
réactives.

## 3.3 Gérer l'historique

| Opération | Effet |
|---|---|
| `conv.reset()` | efface l'historique (garde le prompt système) |
| `conv.undo()` | retire le dernier échange question/réponse |
| `conv.regenerate()` | remplace la dernière réponse par un nouveau tirage |
| `conv.continueReply()` | prolonge la dernière réponse |
| `conv.add(role, texte)` | ajoute un message sans rien générer |
| `conv.history` | la liste des messages (modifiable directement) |

```nim
# fichier : historique.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele)

discard conv.ask("Propose un prénom pour un chat.")
echo "1er tirage : ", conv.history[^1].content
echo "2e tirage  : ", conv.regenerate().text     # autre proposition
conv.undo()                                       # on oublie cet échange
echo "Messages : ", conv.history.len              # 0

# Injecter un faux historique : utile pour reprendre une session ou
# imposer un style par l'exemple (« few-shot »).
conv.add(roleUser, "Traduis : chat")
conv.add(roleAssistant, "cat")
conv.add(roleUser, "Traduis : chien")
conv.add(roleAssistant, "dog")
echo conv.ask("Traduis : oiseau").text            # bird
```

## 3.4 Sauvegarder et reprendre une conversation

```nim
# fichier : sauvegarde.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))

if fileExists("session.json"):
  let conv = newChat(modele)
  conv.loadHistory("session.json")       # système + messages
  echo "Reprise (", conv.history.len, " messages)"
  echo conv.ask("De quoi parlions-nous ?").text
  conv.saveHistory("session.json")
else:
  let conv = newChat(modele, system = "Tu es un coach sportif.")
  echo conv.ask("Je veux courir un 10 km dans 2 mois. Par quoi commencer ?").text
  conv.saveHistory("session.json")
  echo "Session enregistrée ; relancez le programme."
```

Le fichier JSON est lisible et modifiable :

```json
{
  "system": "Tu es un coach sportif.",
  "template": "llama3",
  "messages": [
    {"role": "user", "content": "Je veux courir un 10 km..."},
    {"role": "assistant", "content": "Commence par..."}
  ]
}
```

## 3.5 La fenêtre de contexte

Un modèle ne voit qu'un nombre limité de tokens : la **fenêtre de contexte**.
`newChat(modele, nCtx = 4096)` réserve un cache pour 4096 tokens (Llama 3.2
accepte jusqu’à 131 072, mais chaque token coûte de la mémoire : ~64 Kio pour
Llama 3.2 1B, ~224 Kio pour le 3B).

Que se passe-t-il quand la conversation devient trop longue ?

1. Avant chaque réponse, nimllm vérifie que *prompt + maxTokens* tient dans `nCtx` ;
2. sinon, il retire les **plus anciens** échanges (jamais le prompt système) ;
3. si la dernière question seule est trop longue, `ChatError` est levée.

```nim
# fichier : longue_memoire.nim
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let conv = newChat(modele, nCtx = 512, maxTokens = 100)   # petite fenêtre exprès
conv.system = "Retiens tout ce que je te dis."
for i in 1 .. 15:
  discard conv.ask("Note numéro " & $i & " : j'aime le chiffre " & $(i * 7) & ".")
  echo "tour ", i, " : ", conv.history.len, " messages gardés, ", conv.stats
```

Pour garder une mémoire à long terme malgré tout, deux techniques :

* **résumer** périodiquement l'historique et le placer dans le prompt système ;
* stocker les informations et les **retrouver** au besoin (RAG, chapitre 7).

```nim
# fichier : memoire_resumee.nim
## Quand l'historique dépasse 10 messages, on le remplace par un résumé.
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let systemeDeBase = "Tu es un assistant personnel."
let conv = newChat(modele, system = systemeDeBase)

proc compacter(c: Chat) =
  if c.history.len < 10: return
  var transcript = ""
  for m in c.history: transcript.add $m.role & " : " & m.content & "\n"
  let resumeur = newChat(c.model, sampling = greedySampling(), maxTokens = 200)
  let resume = resumeur.ask("Résume les faits importants de cette conversation " &
                            "en une liste courte :\n" & transcript).text
  c.system = systemeDeBase & "\n\nCe que tu sais déjà de l'utilisateur :\n" & resume
  c.reset()

for msg in ["Je m'appelle Paul.", "J'habite à Rennes.", "J'ai deux chats.",
            "Je travaille comme infirmier.", "Mon plat préféré est la galette.",
            "Comment je m'appelle et où j'habite ?"]:
  echo "> ", msg
  echo conv.ask(msg).text
  compacter(conv)
```

## 3.6 Plusieurs conversations en parallèle

Les poids sont partagés : chaque `Chat` n'ajoute que son propre cache. Vous pouvez
donc tenir plusieurs conversations indépendantes avec un seul modèle chargé.

```nim
# fichier : deux_personnages.nim
## Deux personnages dialoguent entre eux.
import std/os
import nimllm

let modele = loadModel(getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
let pirate = newChat(modele, system = "Tu es un pirate bourru. Une phrase par réplique.", maxTokens = 60)
let robot = newChat(modele, system = "Tu es un robot très poli. Une phrase par réplique.", maxTokens = 60)

var replique = "Bonjour, qui êtes-vous ?"
for tour in 1 .. 3:
  replique = pirate.ask(replique).text
  echo "Pirate : ", replique
  replique = robot.ask(replique).text
  echo "Robot  : ", replique
```

## 3.7 Exercices

1. Écrivez un correcteur d'orthographe : prompt système « Corrige le texte sans
   commentaire » et `greedySampling()`.
2. Ajoutez à `sauvegarde.nim` une commande pour effacer la session.
3. Mesurez la vitesse du 2e tour d'une conversation (`r.seconds`) avec et sans
   `conv.reset()` entre les tours : observez l'effet du cache.

**Suite :** [04 — Réglages de la génération](04-reglages-generation.md)
