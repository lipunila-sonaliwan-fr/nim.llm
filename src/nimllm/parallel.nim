# CC BY-NC-SA 4.0 - jean-marc "jihem" quere 2026
# nimllm/parallel — pool de threads minimaliste (aucune dépendance externe).
#
# Toutes les opérations lourdes (produits matriciels, attention...) passent par
# `parallelFor`, qui découpe un intervalle `0..<n` en morceaux répartis sur
# les threads du pool. Le thread appelant participe lui aussi au calcul.
#
# Compilation : Nim 2.x active `--threads:on` par défaut.

import std/[locks, atomics, cpuinfo]

type
  TaskFn* = proc (ctx: pointer; first, last, worker: int) {.nimcall, gcsafe.}
    # Fonction exécutée sur l'intervalle `first..<last` par le thread `worker`
    # (0 = thread appelant, 1..N-1 = threads du pool).

var
  gLock: Lock
  gCond: Cond
  gWorkers: seq[Thread[int]]
  gNumThreads = 1
  gFn: TaskFn
  gCtx: pointer
  gN: int
  gChunk: int
  gGen: Atomic[int]
  gNext: Atomic[int]
  gDone: Atomic[int]
  gStop: Atomic[bool]
  gBusy: Atomic[bool]
  gInit = false

proc runChunks(worker: int) {.gcsafe.} =
  {.cast(gcsafe).}:
    let n = gN
    let chunk = gChunk
    let fn = gFn
    let ctx = gCtx
    while true:
      let s = gNext.fetchAdd(chunk)
      if s >= n: break
      fn(ctx, s, min(s + chunk, n), worker)

proc workerLoop(id: int) {.thread.} =
  {.cast(gcsafe).}:
    var seen = 0
    while true:
      # attente active courte puis attente sur la condition.
      var spins = 0
      while gGen.load(moAcquire) == seen and not gStop.load(moRelaxed) and spins < 20000:
        cpuRelax()
        inc spins
      if gGen.load(moAcquire) == seen and not gStop.load(moRelaxed):
        acquire(gLock)
        while gGen.load(moAcquire) == seen and not gStop.load(moRelaxed):
          wait(gCond, gLock)
        release(gLock)
      if gStop.load(moRelaxed): break
      seen = gGen.load(moAcquire)
      runChunks(id)
      discard gDone.fetchAdd(1, moRelease)

proc shutdownPool*() =
  # Arrête les threads du pool (appelé automatiquement par `setThreads`).
  if gWorkers.len > 0:
    gStop.store(true)
    acquire(gLock)
    broadcast(gCond)
    release(gLock)
    joinThreads(gWorkers)
    gWorkers.setLen(0)
    gStop.store(false)

proc setThreads*(n: int = 0) =
  # Fixe le nombre de threads de calcul. `n <= 0` : nombre de cœurs détectés.
  if not gInit:
    initLock(gLock); initCond(gCond); gInit = true
  shutdownPool()
  gNumThreads = if n <= 0: max(1, countProcessors()) else: n
  gGen.store(0)
  gWorkers.setLen(gNumThreads - 1)
  for i in 0 ..< gWorkers.len:
    createThread(gWorkers[i], workerLoop, i + 1)

proc numThreads*(): int =
  # Nombre de threads actuellement utilisés.
  if not gInit: setThreads()
  gNumThreads

proc parallelFor*(n: int; fn: TaskFn; ctx: pointer; minChunk = 1) =
  # Exécute `fn` sur `0..<n` en parallèle. Les appels imbriqués s'exécutent
  # séquentiellement (sécurité).
  if n <= 0: return
  if not gInit: setThreads()
  if gNumThreads == 1 or n <= minChunk or gBusy.load(moAcquire):
    fn(ctx, 0, n, 0)
    return
  gBusy.store(true, moRelease)
  gFn = fn
  gCtx = ctx
  gN = n
  gChunk = max(minChunk, n div (gNumThreads * 4))
  gNext.store(0, moRelease)
  gDone.store(0, moRelease)
  acquire(gLock)
  discard gGen.fetchAdd(1, moRelease)
  broadcast(gCond)
  release(gLock)
  runChunks(0)
  let expected = gNumThreads - 1
  while gDone.load(moAcquire) < expected:
    cpuRelax()
  gBusy.store(false, moRelease)
