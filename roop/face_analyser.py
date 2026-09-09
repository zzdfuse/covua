from typing import Any
import insightface

import roop.globals
from roop.typing import Frame

FACE_ANALYSER = None


def _build_providers():
    providers = []
    for p in roop.globals.execution_providers:
        if p == 'CUDAExecutionProvider':
            providers.append((p, {
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'do_copy_in_default_stream': True,
                'gpu_mem_limit': 3 * 1024 * 1024 * 1024,
            }))
        else:
            providers.append(p)
    return providers


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=_build_providers())
        # 320x320 is ~4x faster than 640x640 with negligible quality loss for typical video frames.
        # Use 640 only if faces are very small in frame (e.g. crowd shots).
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(320, 320))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
