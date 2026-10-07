# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# Exemple 09 — Recevoir la réponse sous forme de fichier audio (WAV).
#
#   nim c -r examples/ex09_audio.nim modele.gguf
#
# La synthèse vocale intégrée est un synthétiseur par formants : la voix est
# robotique mais ne nécessite aucun modèle ni bibliothèque externe.

import std/os
import nimllm

let chemin = if paramCount() >= 1: paramStr(1)
             else: getEnv("NIMLLM_MODELE", "modeles/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
let modele = loadModel(chemin)
let conv = newChat(modele, maxTokens = 150)

# 1. Réponse parlée.
conv.voicePitch = 110          # hauteur de la voix (Hz) : ~110 grave, ~200 aiguë
let r = conv.ask("Donne-moi la météo idéale pour un pique-nique, en deux phrases.",
                 format = ofAudio, outPath = "meteo.wav")
echo "Texte prononcé : ", r.text
echo "Fichier : ", r.files[0], " (", readWavInfo(r.files[0]).durationSec, " s)"

# 2. Synthèse directe, sans LLM.
let voix = speak("Bonjour ! Il est 8 heures et 15 minutes.", lang = "fr",
                 pitch = 120, speed = 0.9)
voix.writeWav("bonjour.wav")
echo "Phonèmes : ", textToPhonemes("Bonjour ! Il est 8 heures.")

let anglais = speak("Hello, this is a speech test.", lang = "en")
anglais.writeWav("hello.wav")

# 3. Assembler plusieurs morceaux.
var tout = speak("Premier message.")
tout.silence(0.5)
tout.concat(speak("Deuxième message."))
tout.writeWav("assemblage.wav")

# 4. Le LLM compose une mélodie.
conv.reset()
let partition = conv.ask("Écris une courte mélodie joyeuse en notation : notes séparées " &
  "par des espaces au format Note+Octave[:durée en temps], R pour un silence. " &
  "Exemple : C4 E4 G4:2 R. Réponds uniquement avec les notes.", maxTokens = 80).text
echo "Partition : ", partition
renderMelody(partition, bpm = 110).writeWav("melodie.wav")
