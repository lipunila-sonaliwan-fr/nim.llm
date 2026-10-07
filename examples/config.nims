# Options de compilation communes aux exemples :
#   nim c -r examples/01_bonjour.nim
switch("path", thisDir() & "/../src")   # trouve `import nimllm` sans installation
switch("define", "release")              # optimisations (indispensable : ~10x plus rapide)
switch("passC", "-march=native")         # instructions SIMD du processeur courant
switch("hints", "off")
