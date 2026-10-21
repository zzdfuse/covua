from typing import Any
import threading
import insightface

import roop.globals
from roop.typing import Frame

FACE_ANALYSER = None
THREAD_LOCK = threading.RLock()


def _build_providers():
    providers = []
    for p in roop.globals.execution_providers:
        if p == 'CUDAExecutionProvider':
            providers.append((p, {
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
            }))
        else:
            providers.append(p)
    return providers


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            providers = list(roop.globals.execution_providers)
            if 'CUDAExecutionProvider' in providers and 'CPUExecutionProvider' not in providers:
                providers.append('CPUExecutionProvider')
            print(f"[ROOP.CORE] Loading FaceAnalysis (buffalo_l: detection + recognition)...", flush=True)
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection', 'recognition'],
                providers=providers
            )
            ctx = 0 if 'CUDAExecutionProvider' in providers else -1
            print(f"[ROOP.CORE] Preparing FaceAnalysis on ctx_id={ctx} (det_size: 320x320)...", flush=True)
            FACE_ANALYSER.prepare(ctx_id=ctx, det_size=(320, 320))
            print(f"[ROOP.CORE] Face analyser ready on ctx_id={ctx}!", flush=True)
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    analyser = get_face_analyser()
    with THREAD_LOCK:
        print("[ROOP.CORE] Running face detection on image...", flush=True)
        faces = analyser.get(frame)
        print(f"[ROOP.CORE] Face detection finished: found {len(faces)} face(s).", flush=True)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except (ValueError, IndexError):
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        analyser = get_face_analyser()
        with THREAD_LOCK:
            return analyser.get(frame)
    except IndexError:
        return None
