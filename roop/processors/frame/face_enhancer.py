from typing import Any, List
import cv2
import threading
import gfpgan

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/tuandung/inswapper/resolve/main/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1) # type: ignore[attr-defined]
    return FACE_ENHANCER

def norm_crop2(img, landmark, image_size, enable_padding=True):
	lm = numpy.array(landmark)
	eye_left = lm[0]
	eye_right = lm[1]
	mouth_avg = (lm[3] + lm[4]) * 0.5

	eye_avg = (eye_left + eye_right) * 0.5
	eye_to_eye = eye_right - eye_left
	eye_to_mouth = mouth_avg - eye_avg

	x = eye_to_eye - numpy.flipud(eye_to_mouth) * [-1, 1]
	x /= numpy.hypot(*x)
	rect_scale = 1
	x *= max(numpy.hypot(*eye_to_eye) * 2.0 * rect_scale, numpy.hypot(*eye_to_mouth) * 1.8 * rect_scale)
	y = numpy.flipud(x) * [-1, 1]
	c = eye_avg + eye_to_mouth * 0.1
	quad = numpy.stack([c - x - y, c - x + y, c + x + y, c + x - y])
	qsize = numpy.hypot(*x) * 2

	quad_ori = numpy.copy(quad)
	shrink = int(numpy.floor(qsize / image_size * 0.5))
	if shrink > 1:
		h, w = img.shape[0:2]
		rsize = (int(numpy.rint(float(w) / shrink)), int(numpy.rint(float(h) / shrink)))
		img = cv2.resize(img, rsize, interpolation=cv2.INTER_AREA)
		quad /= shrink
		qsize /= shrink

	h, w = img.shape[0:2]
	border = max(int(numpy.rint(qsize * 0.1)), 3)
	crop = (int(numpy.floor(min(quad[:, 0]))), int(numpy.floor(min(quad[:, 1]))), int(numpy.ceil(max(quad[:, 0]))),
			int(numpy.ceil(max(quad[:, 1]))))
	crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, w), min(crop[3] + border, h))
	if crop[2] - crop[0] < w or crop[3] - crop[1] < h:
		img = img[crop[1]:crop[3], crop[0]:crop[2], :]
		quad -= crop[0:2]

	h, w = img.shape[0:2]
	pad = (int(numpy.floor(min(quad[:, 0]))), int(numpy.floor(min(quad[:, 1]))), int(numpy.ceil(max(quad[:, 0]))),
		   int(numpy.ceil(max(quad[:, 1]))))
	pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - w + border, 0), max(pad[3] - h + border, 0))
	if enable_padding and max(pad) > border - 4:
		pad = numpy.maximum(pad, int(numpy.rint(qsize * 0.3)))
		img = numpy.pad(img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
		h, w = img.shape[0:2]
		y, x, _ = numpy.ogrid[:h, :w, :1]
		mask = numpy.maximum(1.0 - numpy.minimum(numpy.float32(x) / pad[0],
										   numpy.float32(w - 1 - x) / pad[2]),
						  1.0 - numpy.minimum(numpy.float32(y) / pad[1],
										   numpy.float32(h - 1 - y) / pad[3]))
		blur = int(qsize * 0.02)
		if blur % 2 == 0:
			blur += 1
		blur_img = cv2.boxFilter(img, 0, ksize=(blur, blur))

		img = img.astype('float32')
		img += (blur_img - img) * numpy.clip(mask * 3.0 + 1.0, 0.0, 1.0)
		img += (numpy.median(img, axis=(0, 1)) - img) * numpy.clip(mask, 0.0, 1.0)
		img = numpy.clip(img, 0, 255)  # float32, [0, 255]
		quad += pad[:2]

	dst_h, dst_w = image_size, image_size
	template = numpy.array([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]])
	affine_matrix = cv2.estimateAffinePartial2D(quad, template, method=cv2.LMEDS)[0]
	cropped_face = cv2.warpAffine(img, affine_matrix, (dst_w, dst_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray
	affine_matrix = cv2.estimateAffinePartial2D(quad_ori, numpy.array([[0, 0], [0, image_size], [dst_w, dst_h], [dst_w, 0]]), method=cv2.LMEDS)[0]

	return cropped_face, affine_matrix
    
def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
	face_enhancer = get_face_enhancer()
	face_size = 512
	temp_face, matrix = norm_crop2(temp_frame, target_face['kps'], face_size)
	temp_face = temp_face.astype(numpy.float32)[:,:,::-1] / 255.0
	temp_face = (temp_face - 0.5) / 0.5
	temp_face = numpy.expand_dims(temp_face.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
    with THREAD_SEMAPHORE:
        _, _, temp_frame = get_face_enhancer().enhance(
            temp_frame,
            paste_back=True
        )
    return temp_frame

def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
