> © 2026 Jean-Marc Quéré, sonaliwan.fr - Linguistique & Technologies<br/>
Tous droits réservés<br/>
SIRET : 123 456 789 00012<br/>
Licence : CC BY-NC-SA 4.0<br/>

```
  ___    _              ____
 | \ \  (_)_ __ ___    / / /\/\
 | |\ \ | | '_ ` _ \  / / /    \
 | | \ \| | | | | | |/ / / /\/\ \
 |_|  \_\_|_| |_| |_/_/_/\/    \/1.0.0#
```

# Documentation de nimllm

Cette documentation se lit dans l'ordre : chaque chapitre introduit quelques notions
nouvelles et s'appuie sur les précédents. Chaque chapitre contient des programmes
**complets** que vous pouvez copier dans un fichier `.nim` et exécuter tels quels.

## Parcours

| Niveau | Chapitre | Vous saurez… |
|---|---|---|
| Débutant | [01 — Installation](01-installation.md) | installer Nim, compiler, obtenir un modèle GGUF |
| Débutant | [02 — Premiers pas](02-premiers-pas.md) | poser une question, afficher la réponse en direct |
| Débutant | [03 — Conversation et contexte](03-conversation-et-contexte.md) | définir un rôle, dialoguer, sauvegarder, gérer la mémoire |
| Intermédiaire | [04 — Réglages de la génération](04-reglages-generation.md) | contrôler créativité, longueur, répétitions, reproductibilité |
| Intermédiaire | [05 — Pièces jointes](05-pieces-jointes.md) | joindre texte, CSV, code, PDF, images, sons |
| Intermédiaire | [06 — Formats de réponse](06-formats-de-reponse.md) | obtenir du JSON, une image, un fichier audio, un fichier |
| Intermédiaire | [07 — Cas pratiques](07-cas-pratiques.md) | RAG, résumé de longs documents, agents à outils, classification |
| Avancé | [08 — Sous le capot](08-sous-le-capot.md) | tokens, logits, cache KV, gabarits, embeddings, perplexité |
| Avancé | [09 — Bases de l'apprentissage](09-apprentissage-bases.md) | tenseurs, gradients, optimiseurs, réseaux de neurones |
| Expert | [10 — Ajuster un modèle avec LoRA](10-ajuster-avec-lora.md) | spécialiser Llama 3.2 sur vos données |
| Expert | [11 — Créer un nouveau modèle](11-creer-un-modele.md) | tokeniseur, architecture, pré-entraînement, dialogue, export |
| Expert | [12 — GGUF et quantification](12-gguf-et-quantification.md) | inspecter, écrire, quantifier des fichiers de modèle |
| Tous | [13 — Performances et dépannage](13-performances-et-depannage.md) | aller plus vite, comprendre les erreurs |
| Tous | [14 — Référence de l'API](14-reference-api.md) | toutes les fonctions publiques |

## Conventions

* Les programmes commencent par un commentaire `# fichier : nom.nim`.
* Le chemin du modèle est lu sur la ligne de commande ou dans la variable
  d'environnement `NIMLLM_MODELE` :

  ```sh
  export NIMLLM_MODELE=$HOME/modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  nim c -r mon_programme.nim
  ```

* Compilez **toujours** avec `-d:release` (ou utilisez le `config.nims` fourni) :
  en mode debug, le calcul est 10 à 30 fois plus lent.

## Vocabulaire minimal

| Terme | Sens |
|---|---|
| **LLM** | *Large Language Model* : réseau de neurones qui prédit la suite d'un texte |
| **token** | morceau de texte (mot, partie de mot, ponctuation) manipulé par le modèle |
| **contexte** | ensemble des tokens que le modèle « voit » (prompt + historique + réponse) |
| **prompt système** | consigne permanente qui définit le rôle et le comportement du modèle |
| **GGUF** | format de fichier des modèles (poids + tokeniseur + métadonnées) |
| **quantification** | compression des poids (ex. 4 bits au lieu de 32) |
| **logits** | scores bruts attribués par le modèle à chaque token possible |
| **échantillonnage** | façon de choisir le token suivant à partir des logits |
| **LoRA** | technique d'ajustement léger : on n'entraîne que de petites matrices ajoutées |
