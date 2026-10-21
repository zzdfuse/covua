#!/usr/bin/env python3
"""
Telegram Bot for Automated Face Swapping
Monitors Telegram channels for face images and videos, processes them using roop,
and manages output through Google Sheets and forum topics.

Requirements:
    pip install -r requirements.txt  # or requirements-colab.txt for Colab
    pip install -r requirements-telegram-bot.txt

Dependencies:
    - telethon: Telegram client library
    - gspread: Google Sheets API wrapper
    - roop: Face swapping engine (from main requirements)
    
Environment:
    - Requires /content/ssclient.session file for Telegram authentication
    - Requires Google Service Account credentials (embedded in script)
    - Designed to run on Google Colab with CUDA support
"""

import asyncio
import os
import sys
from time import sleep
import logging

# Must be set before any torch/onnxruntime import — prevents CPU OpenMP thread explosion
# which starves the GPU by keeping CPU cores busy with scheduling overhead.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')   # faster CUDA init
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')     # silence TensorFlow cudart notices
from datetime import datetime
from asyncio import Queue

from telethon import TelegramClient, events
from telethon.tl.types import InputMessagesFilterPhotos, InputMessagesFilterVideo, DocumentAttributeVideo, DocumentAttributeImageSize
from telethon.tl.functions.channels import CreateChannelRequest

# Telethon compatibility: Telethon 1.42+ (used on Google Colab) moved CreateForumTopicRequest to messages
try:
    from telethon.tl.functions.messages import CreateForumTopicRequest
except ImportError:
    try:
        from telethon.tl.functions.channels import CreateForumTopicRequest
    except ImportError:
        CreateForumTopicRequest = None
from telethon.events import MessageDeleted
import gspread

# Roop is invoked as a subprocess per render so each process gets isolated globals.
# Concurrent renders share the GPU safely — onnxruntime handles multi-process CUDA access.
# run.py lives alongside telegram_bot.py; ROOP_PATH only needed as a fallback for
# external installs where run.py and roop/ are elsewhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOP_PATH = os.getenv('ROOP_PATH', SCRIPT_DIR)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
)
logger = logging.getLogger(__name__)

def log_separator(title=""):
    """Create a visual separator in logs"""
    if title:
        logger.info(f"{'='*20} {title} {'='*20}")
    else:
        logger.info("="*50)

# ============================================================================
# PREREQUISITE & CUDA CAPABILITY CHECK (STEP 1)
# ============================================================================
_system_check_cache = None

def check_cuda_and_prerequisites(silent=False, force=False):
    """
    Step 1 Pre-flight check:
    Inspects CUDA readiness for both Video (ONNX Runtime / Roop) and Image (PyTorch / SD inpainting)
    processing, verifies system drivers, hardware, and external tools, and pinpoints exact reasons
    for any missing prerequisites.
    """
    global _system_check_cache
    if _system_check_cache is not None and not force:
        return _system_check_cache

    import subprocess
    import shutil
    import platform

    results = {}
    issues = []

    # 1. System & NVIDIA Driver (Hardware layer)
    smi_path = shutil.which('nvidia-smi')
    if smi_path:
        try:
            res = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_lines = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
                results['nvidia_gpu'] = ('OK', '; '.join(gpu_lines))
            else:
                err_msg = res.stderr.strip()[:60] if res.stderr else "empty response"
                results['nvidia_gpu'] = ('WARN', f'nvidia-smi error: {err_msg}')
                issues.append(('NVIDIA Driver', 'nvidia-smi returned an error.', 'Verify NVIDIA driver status.'))
        except Exception as e:
            results['nvidia_gpu'] = ('WARN', f'Failed to query nvidia-smi: {e}')
            issues.append(('NVIDIA Driver', str(e), 'Verify nvidia-smi command.'))
    else:
        plat = platform.system()
        if plat == 'Darwin':
            reason = 'Running on macOS. Apple hardware does not support NVIDIA CUDA.'
            fix = 'To use CUDA acceleration, run on Linux or Google Colab with an NVIDIA GPU runtime.'
        else:
            reason = 'nvidia-smi not found in PATH. NVIDIA driver is not installed or no NVIDIA GPU is present.'
            fix = 'If on Google Colab, enable GPU runtime (Runtime > Change runtime type > T4 GPU). If on Linux, install NVIDIA driver (e.g. sudo apt install -y nvidia-driver-535).'
        results['nvidia_gpu'] = ('FAIL', 'No NVIDIA GPU / driver detected')
        issues.append(('NVIDIA Driver & Hardware', reason, fix))

    # 2. PyTorch CUDA (Image Inpainting / Cloth Removal / Face Analysis)
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_ver = torch.version.cuda or 'Unknown'
            cudnn_ver = torch.backends.cudnn.version() or 'Unknown'
            results['torch_cuda'] = ('OK', f'CUDA {cuda_ver}, cuDNN {cudnn_ver}, {dev_name} ({mem_gb:.1f} GB)')
        else:
            results['torch_cuda'] = ('FAIL', f'PyTorch {torch.__version__} (CUDA unavailable)')
            if torch.version.cuda is None:
                r = 'PyTorch was installed as a CPU-only build (torch.version.cuda is None).'
                f = 'Reinstall PyTorch with CUDA: pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118'
            elif torch.cuda.device_count() == 0:
                r = f'PyTorch has CUDA build ({torch.version.cuda}), but detected 0 GPUs.'
                f = 'Verify NVIDIA driver is loaded and CUDA_VISIBLE_DEVICES is not hiding GPUs.'
            else:
                r = 'PyTorch CUDA initialization failed.'
                f = 'Check NVIDIA driver compatibility with the installed CUDA runtime.'
            issues.append(('PyTorch CUDA (Image Inpainting / SD)', r, f))
    except Exception as e:
        results['torch_cuda'] = ('FAIL', f'torch error: {str(e)[:35]}')
        issues.append(('PyTorch', f'Error loading torch: {e}', 'Install torch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118'))

    # 3. ONNX Runtime CUDA (Video Face Swapping / Roop)
    try:
        import importlib.metadata
        cpu_installed = False
        gpu_installed = False
        try:
            importlib.metadata.version('onnxruntime')
            cpu_installed = True
        except Exception:
            pass
        try:
            importlib.metadata.version('onnxruntime-gpu')
            gpu_installed = True
        except Exception:
            pass

        if cpu_installed and gpu_installed:
            results['ort_cuda'] = ('FAIL', 'FATAL: Both onnxruntime & onnxruntime-gpu installed!')
            r = 'Both CPU onnxruntime and onnxruntime-gpu are installed in the same environment. CPU binaries overwrite GPU shared libraries, silently disabling CUDA.'
            f = 'Run: pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu'
            issues.append(('ONNX Runtime Package Conflict', r, f))
        else:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in providers:
                results['ort_cuda'] = ('FAIL', f'Only fallback providers: {providers}')
                r = 'CUDAExecutionProvider is missing from ONNX Runtime.'
                f = 'Run: pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu'
                issues.append(('ONNX Runtime (Video Face Swapping / Roop)', r, f))
            else:
                # Live validation test: Create an actual in-memory InferenceSession using CUDAExecutionProvider
                try:
                    import onnx
                    from onnx import helper, TensorProto
                    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1])
                    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1])
                    node = helper.make_node('Identity', ['X'], ['Y'])
                    graph = helper.make_graph([node], 'test', [X], [Y])
                    model = helper.make_model(graph)
                    model_bytes = model.SerializeToString()
                    
                    test_sess = ort.InferenceSession(model_bytes, providers=['CUDAExecutionProvider'])
                    active = test_sess.get_providers()
                    if 'CUDAExecutionProvider' in active:
                        results['ort_cuda'] = ('OK', f'CUDA verified (ORT v{ort.__version__})')
                    else:
                        results['ort_cuda'] = ('FAIL', f'Fell back to: {active}')
                        issues.append(('ONNX Runtime CUDA Fallback', 'CUDA provider failed to activate during session creation.', 'Run: pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu'))
                except Exception as sess_err:
                    err_str = str(sess_err)
                    results['ort_cuda'] = ('FAIL', f'CUDA session failed: {err_str[:35]}')
                    r = f'ONNX Runtime CUDA failed to initialize: {err_str}'
                    f = 'Run: pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu'
                    issues.append(('ONNX Runtime CUDA Initialization Error', r, f))
    except Exception as e:
        err_msg = str(e)
        results['ort_cuda'] = ('FAIL', f'onnxruntime error: {err_msg[:35]}')
        if 'numpy' in err_msg.lower() or 'multiarray' in err_msg.lower():
            r = f'onnxruntime failed due to NumPy version incompatibility: {err_msg}'
            f = "Downgrade NumPy: pip install 'numpy<2.0.0'"
        else:
            r = f'onnxruntime package error: {err_msg}'
            f = 'Reinstall onnxruntime-gpu: pip install onnxruntime-gpu'
        issues.append(('ONNX Runtime (Video Face Swapping / Roop)', r, f))

    # 4. FFmpeg Binary (Video decoding, encoding, audio remuxing)
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        results['ffmpeg'] = ('OK', ffmpeg_path)
    else:
        results['ffmpeg'] = ('FAIL', 'ffmpeg not found in PATH')
        issues.append(('FFmpeg', 'ffmpeg executable is missing.', 'Install: apt-get install -y ffmpeg (Linux) or brew install ffmpeg (macOS)'))

    # 5. Roop Face Swap Model (inswapper_128.onnx)
    model_candidates = [
        os.path.join(SCRIPT_DIR, 'roop', 'models', 'inswapper_128.onnx'),
        os.path.join(SCRIPT_DIR, 'models', 'inswapper_128.onnx'),
        os.path.expanduser('~/.roop/inswapper_128.onnx'),
        '/content/inswapper_128.onnx'
    ]
    found_model = None
    for p in model_candidates:
        if os.path.exists(p):
            sz = os.path.getsize(p) / (1024 * 1024)
            found_model = f'{os.path.basename(p)} ({sz:.1f} MB)'
            break
    if found_model:
        results['roop_model'] = ('OK', found_model)
    else:
        results['roop_model'] = ('INFO', 'Not cached yet (will auto-download on first render)')

    # 6. InsightFace Face Detector / Analyser (buffalo_l)
    insightface_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
    buffalo_det = os.path.join(insightface_dir, 'det_10g.onnx')
    if os.path.exists(buffalo_det):
        results['face_analyser'] = ('OK', 'buffalo_l (Face Analysis cached)')
    else:
        try:
            logger.info("📥 Downloading InsightFace buffalo_l models (~300MB) for face detection...")
            import urllib.request
            import zipfile
            import tempfile
            os.makedirs(insightface_dir, exist_ok=True)
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                tmp_zip = tmp.name
            urllib.request.urlretrieve(url, tmp_zip)
            with zipfile.ZipFile(tmp_zip, 'r') as zf:
                zf.extractall(insightface_dir)
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)
            results['face_analyser'] = ('OK', 'buffalo_l (Downloaded & ready)')
        except Exception as e:
            results['face_analyser'] = ('INFO', f'Not cached ({e})')

    video_cuda = results.get('ort_cuda', ('FAIL', ''))[0] == 'OK'
    image_cuda = results.get('torch_cuda', ('FAIL', ''))[0] == 'OK'
    ffmpeg_ok = results.get('ffmpeg', ('FAIL', ''))[0] == 'OK'

    # Build display banner
    banner = []
    banner.append("=" * 82)
    banner.append("🔍 STEP 1: HARDWARE & CUDA ACCELERATION PREREQUISITE CHECK")
    banner.append("=" * 82)
    fmt = "| %-26s | %-6s | %-42s |"
    banner.append(fmt % ('Component', 'Status', 'Diagnostic Details'))
    banner.append("+" + "-" * 28 + "+" + "-" * 8 + "+" + "-" * 44 + "+")

    status_icon = {'OK': '✅ OK', 'FAIL': '❌ FAIL', 'WARN': '⚠️ WARN', 'INFO': 'ℹ️ INFO'}
    labels = {
        'nvidia_gpu': 'NVIDIA Driver & Hardware',
        'torch_cuda': 'PyTorch CUDA (Image/SD)',
        'ort_cuda': 'ONNX Runtime (Video/Roop)',
        'ffmpeg': 'FFmpeg Binary',
        'roop_model': 'Roop Face Swap Model',
        'face_analyser': 'InsightFace (buffalo_l)',
    }

    for key, label in labels.items():
        st, det = results.get(key, ('UNKNOWN', ''))
        icon = status_icon.get(st, st)
        det_disp = det if len(det) <= 42 else det[:39] + '...'
        banner.append(fmt % (label, icon, det_disp))
    banner.append("+" + "-" * 28 + "+" + "-" * 8 + "+" + "-" * 44 + "+")

    banner.append("\n📊 ACCELERATION READINESS:")
    if video_cuda and ffmpeg_ok:
        v_msg = "✅ CUDA ACCELERATED (Full GPU Support)"
    elif ffmpeg_ok:
        v_msg = "⚠️ CPU FALLBACK (SLOW - ONNX Runtime has no CUDA; rendering will run on CPU)"
    else:
        v_msg = "❌ UNAVAILABLE (Missing FFmpeg or ONNX Runtime)"

    if image_cuda:
        i_msg = "✅ CUDA ACCELERATED (Full GPU Support)"
    else:
        i_msg = "❌ UNAVAILABLE (SD inpainting strictly requires CUDA)"

    banner.append(f"   🎥 Video Face Swap (Roop):  {v_msg}")
    banner.append(f"   🖼️ Image Inpaint (SD):      {i_msg}")

    if issues:
        banner.append("\n⚠️ PREREQUISITE ISSUES & ACTIONABLE FIXES:")
        for idx, (comp, reason, fix) in enumerate(issues, 1):
            banner.append(f"   [{idx}] {comp}:")
            banner.append(f"       • Reason: {reason}")
            banner.append(f"       • Fix:    {fix}")
    else:
        banner.append("\n🎉 All prerequisites met! System is fully optimized for GPU rendering.")
    banner.append("=" * 82)

    output_str = "\n".join(banner)
    if not silent:
        print(output_str)
        for line in banner:
            logger.info(line)

    _system_check_cache = {
        'video_cuda': video_cuda,
        'image_cuda': image_cuda,
        'ffmpeg_ok': ffmpeg_ok,
        'results': results,
        'issues': issues,
        'banner_str': output_str
    }
    return _system_check_cache

# Run prerequisite & CUDA check as the very first step when starting python telegram_bot.py
if __name__ == "__main__":
    check_cuda_and_prerequisites()
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--check", "--check-cuda", "-c", "--prereq"):
        sys.exit(0)

# ============================================================================
# SD INPAINTING CONFIGURATION (etcd thread — cloth removal)
# ============================================================================
_sd_pipe = None
_seg_model = None

SD_MODEL_ID    = os.getenv('SD_MODEL_ID', 'Uminosachi/realisticVisionV51_v51VAE-inpainting')
SD_PROMPT      = (
    'RAW photo, (masterpiece:1.3), (best quality:1.3), (ultra-realistic:1.2), '
    'photorealistic, hyperrealism, DSLR, 85mm lens, f/1.8, soft bokeh, '
    '(bare skin:1.4), (nude:1.3), (no clothing:1.3), '
    'Vietnamese woman, Southeast Asian female, (warm golden skin tone:1.2), '
    '(smooth flawless skin:1.2), realistic skin texture, visible skin pores, '
    'subsurface scattering, natural skin sheen, '
    '(slim petite figure:1.2), toned athletic body, natural feminine curves, '
    '(symmetrical natural breasts:1.3), (proportional chest:1.2), correct anatomy, '
    'soft studio lighting, natural shadows, 8k uhd, sharp focus, film grain'
)
SD_NEGATIVE    = (
    '(worst quality:2), (low quality:2), (normal quality:2), (bad quality:2), '
    'lowres, jpeg artifacts, signature, watermark, username, blurry, '
    '(bad anatomy:1.8), (deformed:1.6), (malformed:1.6), (mutated:1.6), '
    '(extra limbs:1.5), (extra arms:1.5), (extra legs:1.5), (missing limbs:1.5), '
    '(fused fingers:1.5), (too many fingers:1.5), (extra fingers:1.5), '
    '(uneven breasts:1.6), (asymmetrical breasts:1.6), (deformed breasts:1.6), '
    '(oversized breasts:1.4), (bad proportions:1.5), long neck, cross-eyed, '
    'clothing, shirt, t-shirt, jacket, coat, vest, bra, underwear, fabric, garment, dressed, covered, '
    '(plastic skin:1.4), (doll:1.4), (mannequin:1.4), (3d render:1.3), '
    'cartoon, anime, painting, drawing, sketch, illustration, '
    'caucasian, western features, dark skin, tan lines, '
    'ugly, duplicate, morbid, mutilated, out of frame'
)
SD_GUIDANCE    = float(os.getenv('SD_GUIDANCE', '7.5'))
SD_STRENGTH    = float(os.getenv('SD_STRENGTH', '0.78'))
SD_STEPS       = int(os.getenv('SD_STEPS', '75'))
SD_TARGET_SIZE = int(os.getenv('SD_TARGET_SIZE', '512'))
SD_SEED        = int(os.getenv('SD_SEED', '42'))

# ============================================================================
# VIDEO RENDERING QUEUE
# ============================================================================
# Global queue for video rendering tasks - ensures one video renders at a time
# while keeping the bot responsive to other operations
render_queue = Queue()
render_worker_tasks = []

# ============================================================================
# TELEGRAM CONFIGURATION
# ============================================================================
api_id = 25324831
api_hash = '3ae696f7cf3fcb591da4b8c9cda9c41e'

# Session file path - use local path when not on Colab
SESSION_PATH = os.getenv('TELEGRAM_SESSION_PATH', './ssclient.session')
client = TelegramClient(SESSION_PATH, api_id, api_hash)

# Personal session for folder management (user account, not bot)
PERSONAL_SESSION_PATH = os.getenv('TELEGRAM_PERSONAL_SESSION_PATH', './sspersonal.session')
personal_client = TelegramClient(PERSONAL_SESSION_PATH, api_id, api_hash)

entity_map = {
    "group": {
        "id": 2192019759,
        "threads": {
            "inputimage": 2,
            "inputvideo": 3,
            "output": 4,
            "etcd": 375
        }
    },
    "output_chat_id": {
        "id": 2326968720,
        "threads": {}
    },
    "user_id": {
        "id": 5177338966,
        "threads": {}
    },
    "input_chat_id": {
        "id": 2354814383,
        "threads": {
            "faces": 2,
            "vid": 3,
            "etcd": 7
        }
    }
}

def get_entity_id(entity_name):
    return entity_map[entity_name]["id"]

thread_map = {
    "inputimage": 2,
    "inputvideo": 3,
    "output": 4,
    "etcd": 375
}

entity_list = {}

# Directory paths - use local paths when not on Colab
BASE_DIR = os.getenv('BOT_BASE_DIR', '.')
os_dir = {
    "input_image": os.path.join(BASE_DIR, "input_image"),
    "input_video": os.path.join(BASE_DIR, "input_video"),
    "output": os.path.join(BASE_DIR, "output")
}

# Create directories if they don't exist
for dir_path in os_dir.values():
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# GOOGLE SHEETS CONFIGURATION
# ============================================================================
SHEET_ID = "1ic3oeMukAoQKVRkaJAs55f9bTpAACvzXmBLKc_1Kq2A"

# Load service account credentials from file
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', './service_account.json')

if not os.path.exists(SERVICE_ACCOUNT_FILE):
    logger.error(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
    logger.error(f"   Please create a service account JSON file from Google Cloud Console")
    logger.error(f"   or set GOOGLE_SERVICE_ACCOUNT_FILE environment variable")
    sys.exit(1)

logger.info(f"📋 Loading Google Sheets credentials from: {SERVICE_ACCOUNT_FILE}")
gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
sh = gc.open_by_key(SHEET_ID)
logger.info(f"✅ Connected to Google Sheet: {SHEET_ID}")

# ============================================================================
# GOOGLE SHEETS FUNCTIONS
# ============================================================================
def get_data(worksheet_name):
    return sh.worksheet(worksheet_name).get_all_values()[1:]

def check_topic_exist(image_name, data=None):
    if data is None:
        data = get_data("list_image")
    for i in data:
        if i[1] == image_name and i[3] != "":
            return True, i[4]
    return False, None

def check_image_by_message_id(message_id, data=None):
    """Check if an image entry exists by message_id and return its data"""
    if data is None:
        data = get_data("list_image")
    for i in range(len(data)):
        if data[i][0] == str(message_id):
            # Old structure: [message_id, message_text, output_chat_id, topic_id, old_name]
            # New structure: [message_id, message_text, output_chat_id, topic_id, old_name, channel_id]
            # Return: (exists, row_index, message_text, output_chat_id, topic_id, channel_id, old_name)
            channel_id = data[i][5] if len(data[i]) > 5 else ""  # Column 6 (index 5)
            old_name = data[i][4] if len(data[i]) > 4 else data[i][1]  # Column 5 (index 4)
            return True, i, data[i][1], data[i][2], data[i][3], channel_id, old_name
    return False, None, None, None, None, None, None

def is_video_message(message):
    if not message:
        return False
    if getattr(message, 'video', None):
        return True
    doc = getattr(message, 'document', None)
    if doc:
        mime = getattr(doc, 'mime_type', '') or ''
        if mime.startswith('video/'):
            return True
        for attr in getattr(doc, 'attributes', []) or []:
            if isinstance(attr, DocumentAttributeVideo):
                return True
    return False

def is_photo_message(message):
    if not message:
        return False
    if getattr(message, 'photo', None):
        return True
    doc = getattr(message, 'document', None)
    if doc:
        mime = getattr(doc, 'mime_type', '') or ''
        if mime.startswith('image/'):
            return True
        for attr in getattr(doc, 'attributes', []) or []:
            if isinstance(attr, DocumentAttributeImageSize):
                return True
    return False

def sanitize_video_name(text):
    if not text:
        return ""
    first_line = text.strip().split("\n")[0].strip()
    words = [w for w in first_line.split() if not w.startswith("http://") and not w.startswith("https://") and not w.startswith("@")]
    clean = " ".join(words).strip().lower()
    clean = clean.strip(".,;:!?_-\"'/\\")
    if len(clean) > 40:
        clean = clean[:40].strip()
    return clean

def check_video_exist(video_id, data=None):
    if data is None:
        data = get_data("list_video")
    for idx, row in enumerate(data):
        if str(row[0]) == str(video_id):
            name = row[1] if len(row) > 1 else ""
            return True, name, idx + 2
    return False, None, None

def get_topic_id(image_id, data=None):
    if data is None:
        data = get_data("list_image")
    for i in data:
        if i[0] == image_id:
            return i[3]
    return None

def check_output_exist(output_id, data=None):
    if data is None:
        data = get_data("list_output")
    for i in data:
        if i[0] == output_id:
            return True, i[1]
    return False, None

def update_topic_id(image_id, image_name, data=None):
    if data is None:
        data = get_data("list_image")
    logger.info(f"📝 Updating topic ID for image: {image_name} → {image_id}")
    for i in range(len(data)):
        if data[i][1] == image_name:
            try:
                sh.worksheet("list_image").update_acell(f"A{i+2}", image_id)
                logger.info(f"✅ Successfully updated topic ID for {image_name}")
                return True
            except Exception as e:
                logger.error(f"❌ Error updating topic ID for {image_name}: {e}")
                return False
    logger.warning(f"⚠️ Image name {image_name} not found in data")
    return True

def delete_image_by_message_id(message_id):
    """Delete image entry from Google Sheet by message ID"""
    logger.info(f"🔍 Checking for image with message ID: {message_id}")
    try:
        data = get_data("list_image")
        for i in range(len(data)):
            if data[i][0] == str(message_id):
                row_number = i + 2
                sh.worksheet("list_image").delete_rows(row_number)
                logger.info(f"✅ Successfully deleted image '{data[i][1]}' (ID: {message_id}) from sheet")
                return True, data[i][1]
        logger.info(f"ℹ️ Image with message ID {message_id} not found in sheet")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error deleting image with message ID {message_id}: {e}")
        return False, None

def delete_video_by_message_id(message_id):
    """Delete video entry from Google Sheet by message ID"""
    logger.info(f"🔍 Checking for video with message ID: {message_id}")
    try:
        data = get_data("list_video")
        for i in range(len(data)):
            if data[i][0] == str(message_id):
                row_number = i + 2
                sh.worksheet("list_video").delete_rows(row_number)
                logger.info(f"✅ Successfully deleted video '{data[i][1]}' (ID: {message_id}) from sheet")
                return True, data[i][1]
        logger.info(f"ℹ️ Video with message ID {message_id} not found in sheet")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error deleting video with message ID {message_id}: {e}")
        return False, None

def delete_outputs_by_image_id(image_id):
    """Delete all output entries that contain this image ID"""
    logger.info(f"🗑️ Deleting all outputs containing image ID: {image_id}")
    try:
        data = get_data("list_output")
        deleted_count = 0
        deleted_outputs = []

        for i in range(len(data) - 1, -1, -1):
            output_id = data[i][0]
            if output_id.startswith(f"{image_id}_"):
                row_number = i + 2
                sh.worksheet("list_output").delete_rows(row_number)
                deleted_outputs.append(data[i][1])
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"✅ Successfully deleted {deleted_count} output(s) containing image ID {image_id}")
            logger.info(f"   📝 Deleted outputs: {', '.join(deleted_outputs)}")
        else:
            logger.info(f"ℹ️ No outputs found containing image ID {image_id}")

        return deleted_count, deleted_outputs
    except Exception as e:
        logger.error(f"❌ Error deleting outputs for image ID {image_id}: {e}")
        return 0, []

def create_map_user_image():
    data = get_data("list_image")
    map_user_image = {}
    for i in data:
        if i[1] not in map_user_image:
            # Sheet structure: [message_id, message_text, output_chat_id, topic_id, old_name, channel_id]
            channel_id = i[5] if len(i) > 5 else None
            map_user_image[i[1]] = {
                "chat_id": i[2],
                "topic_id": i[3],
                "message_id": i[0],
                "channel_id": channel_id
            }
    return map_user_image

def create_map_video():
    data = get_data("list_video")
    map_video = {}
    for i in data:
        map_video[i[1]] = {
            "message_id": i[0]
        }
    return map_video

# ============================================================================
# TELEGRAM HELPER FUNCTIONS
# ============================================================================
def entity(entity_name):
    if entity_name not in entity_map:
        return None
    if entity_name not in entity_list:
        entity_list[entity_name] = client.get_entity(get_entity_id(entity_name))
    return entity_list[entity_name]

async def download_file(message_id, sub_path=".", ext="jpg"):
    logger.info(f"📥 Downloading file {message_id} to {sub_path}")
    message_id = int(message_id)
    final_path = f"{sub_path}/{message_id}.{ext}"
    
    if os.path.exists(final_path):
        logger.info(f"⚡ File {final_path} already exists, skipping download")
        return final_path

    try:
        message = await client.get_messages(get_entity_id("input_chat_id"), limit=1, ids=message_id)
        logger.info(f"📨 Retrieved message: {message}")
        result = await client.download_media(message, final_path)
        logger.info(f"✅ Successfully downloaded file to {final_path}")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to download file {message_id}: {e}")
        raise

async def render_video(input_image, input_video, output_path):
    """
    Render a face-swap video by spawning roop as a subprocess.
    Each subprocess gets its own copy of roop.globals — no shared-state race conditions
    when multiple renders run concurrently. The GPU is shared safely via CUDA.
    """
    logger.info(f"🎬 Starting video render: {input_video} with image {input_image} → {output_path}")

    import json
    roop_config_file = os.getenv('ROOP_CONFIG_FILE', './roop_config.json')
    cfg = {}
    if os.path.exists(roop_config_file):
        with open(roop_config_file) as f:
            cfg = json.load(f)

    # run.py lives next to telegram_bot.py; fall back to ROOP_PATH for external installs
    run_py = os.path.join(SCRIPT_DIR, 'run.py')
    if not os.path.exists(run_py):
        run_py = os.path.join(ROOP_PATH, 'run.py')

    # Check execution provider: prefer cuda if available, fallback to cpu
    exec_provider = 'cuda'
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            exec_provider = 'cpu'
            logger.warning("⚠️ CUDAExecutionProvider not available in ONNX Runtime; falling back to CPU for Roop rendering")
    except Exception:
        exec_provider = 'cpu'

    cmd = [
        sys.executable, run_py,
        '-s', input_image,
        '-t', input_video,
        '-o', output_path,
        '--execution-provider', exec_provider,
        '--execution-threads', str(cfg.get('execution_threads', 4)),
        '--max-memory', str(cfg.get('max_memory', 6)),
        '--video-encoder', cfg.get('video_encoder', 'libx264'),
        '--video-quality', str(cfg.get('video_quality', 23)),
        '--frame-processor', 'face_swapper',
    ]
    if cfg.get('many_faces', False):
        cmd.append('--many-faces')
    if cfg.get('keep_fps', True):
        cmd.append('--keep-fps')
    if cfg.get('keep_audio', True):
        cmd.append('--keep-audio')

    logger.info(f"🔧 Roop cmd: {' '.join(cmd)}")

    # cwd=SCRIPT_DIR so relative paths (./image, ./video, ./output) resolve correctly.
    # PYTHONPATH includes SCRIPT_DIR so subprocess can `import roop` from the local roop/ package.
    # stdout/stderr intentionally inherit from parent so roop's tqdm progress bars
    # appear in real-time. Error detection falls back to returncode + output file check.
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    env = {
        **os.environ,
        'PYTHONUNBUFFERED': '1',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'OMP_NUM_THREADS': '1',
        'PYTHONPATH': f"{SCRIPT_DIR}:{existing_pythonpath}" if existing_pythonpath else SCRIPT_DIR,
    }
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=SCRIPT_DIR,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output_lines = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded = line.decode('utf-8', errors='replace').rstrip()
        if decoded:
            output_lines.append(decoded)
            if "cudart_stub.cc" not in decoded and "absl::InitializeLog" not in decoded:
                logger.info(f"   [Roop] {decoded}")

    await proc.wait()

    if proc.returncode != 0:
        tail = "\n".join(output_lines[-15:])
        raise Exception(f"Roop subprocess failed with exit code {proc.returncode}:\n{tail}")

    if not os.path.exists(output_path):
        tail = "\n".join(output_lines[-15:])
        raise Exception(f"Roop exited 0 but output file missing: {output_path}\nOutput tail:\n{tail}")

    logger.info(f"✅ Video render completed: {output_path}")

async def process_render_queue():
    """
    Worker that processes video rendering tasks one at a time from the queue
    This ensures only one video renders at a time while keeping bot responsive
    """
    logger.info("🎬 Video render queue worker started")
    while True:
        try:
            # Get next rendering task from queue
            task_data = await render_queue.get()
            
            if task_data is None:  # Shutdown signal
                logger.info("🛑 Render queue worker shutting down")
                break
            
            combo_name = task_data['combo_name']
            input_image = task_data['input_image']
            input_video = task_data['input_video']
            output_path = task_data['output_path']
            output_thread = task_data['output_thread']
            channel_id = task_data['channel_id']
            image_id = task_data['image_id']
            video_id = task_data['video_id']
            
            logger.info(f"🎬 Queue: Processing render for {combo_name}")
            
            try:
                # Render the video (this blocks, but we're in a dedicated worker)
                await render_video(input_image, input_video, output_path)
                logger.info(f"🎬 Video rendering completed: {combo_name}")
                
                # Send to forum topic
                await send_video("output_chat_id", output_thread, output_path, combo_name)
                logger.info(f"✅ Video sent to forum topic successfully: {combo_name}")
                
                # Send to channel if channel_id exists
                if channel_id:
                    await client.send_file(int(channel_id), output_path, caption=combo_name)
                    logger.info(f"✅ Video sent to channel successfully: {combo_name}")
                
                # Mark as successful in sheets
                await send_message("input_chat_id", thread_map["etcd"], f"✅ {combo_name} completed successfully")
                sh.worksheet("list_output").append_row([f"{image_id}_{video_id}", combo_name])
                logger.info(f"✅ Successfully completed: {combo_name}")
                
                # Notify about success
                if 'tracking_message' in task_data and task_data['tracking_message']:
                    await task_data['success_callback'](combo_name)
                
            except Exception as e:
                logger.error(f"❌ Queue: Failed to render {combo_name}: {e}")
                await send_message("group", thread_map["etcd"], f"❌ {combo_name} failed: {str(e)}")
                
                # Notify about failure
                if 'tracking_message' in task_data and task_data['tracking_message']:
                    await task_data['failure_callback'](combo_name, str(e))
            
            finally:
                # Mark task as done
                render_queue.task_done()
                
        except Exception as e:
            logger.error(f"❌ Render queue worker error: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

async def create_forum_topic(topic_name, chat_id):
    logger.info(f"🆕 Creating forum topic: '{topic_name}' in chat {chat_id}")
    if CreateForumTopicRequest is None:
        logger.error("❌ CreateForumTopicRequest is not available in the installed Telethon version")
        return None
    try:
        # Telethon 1.42+ (in messages, used on Colab) uses 'peer'; older versions use 'channel'
        try:
            topic = await client(CreateForumTopicRequest(
                peer=chat_id,
                title=topic_name,
                icon_color=None,  # Optional: can set a color
                icon_emoji_id=None  # Optional: can set an emoji
            ))
        except TypeError:
            topic = await client(CreateForumTopicRequest(
                channel=chat_id,
                title=topic_name,
                icon_color=None,
                icon_emoji_id=None
            ))
        
        # Extract topic ID from response
        # The response structure may vary between versions
        if hasattr(topic, 'updates') and len(topic.updates) > 0:
            topic_id = topic.updates[0].id
        else:
            logger.warning(f"⚠️ Unexpected response structure, trying to extract topic ID")
            # Try to find the topic ID in the response
            topic_id = getattr(topic.updates[0], 'id', None) if hasattr(topic, 'updates') else None
            
        if topic_id:
            logger.info(f"✅ Forum topic created successfully: '{topic_name}' (ID: {topic_id})")
            return topic_id
        else:
            logger.error(f"❌ Could not extract topic ID from response: {topic}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to create forum topic '{topic_name}': {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return None

async def create_separate_chat(chat_name, photo_message=None):
    """
    Create a separate channel for an image
    This allows grouping multiple chats into folders later
    Optionally sets the image as the channel avatar
    """
    logger.info(f"📢 Creating separate channel: '{chat_name}'")
    try:
        # Create a new channel (can be private or public)
        result = await client(CreateChannelRequest(
            title=chat_name,
            about=f"Chat for {chat_name}",
            megagroup=False  # False = channel, True = supergroup
        ))
        
        # Extract channel ID from response
        if hasattr(result, 'chats') and len(result.chats) > 0:
            channel = result.chats[0]
            channel_id = channel.id
            logger.info(f"✅ Channel created successfully: '{chat_name}' (ID: {channel_id})")
            
            # Set the photo as channel avatar if provided
            if photo_message and photo_message.photo:
                try:
                    logger.info(f"🖼️ Setting channel avatar from image...")
                    
                    # Download the photo to a temporary file with proper extension
                    import tempfile
                    temp_photo = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    photo_path = temp_photo.name
                    temp_photo.close()
                    
                    await client.download_media(photo_message, file=photo_path)
                    logger.info(f"📥 Photo downloaded to {photo_path}")
                    
                    # Upload and set as channel photo using InputChatUploadedPhoto
                    from telethon.tl.functions.channels import EditPhotoRequest
                    from telethon.tl.types import InputChatUploadedPhoto
                    
                    uploaded_file = await client.upload_file(photo_path)
                    await client(EditPhotoRequest(
                        channel=channel,
                        photo=InputChatUploadedPhoto(file=uploaded_file)
                    ))
                    
                    # Clean up temp file
                    import os as temp_os
                    temp_os.unlink(photo_path)
                    
                    logger.info(f"✅ Channel avatar set successfully for '{chat_name}'")
                except Exception as avatar_error:
                    logger.warning(f"⚠️ Failed to set channel avatar: {avatar_error}")
                    import traceback
                    logger.warning(f"   Traceback: {traceback.format_exc()}")
                    # Don't fail the whole operation if avatar setting fails
            
            # Add channel to folder
            try:
                await add_channel_to_folder(channel, "ODF")
            except Exception as folder_error:
                logger.warning(f"⚠️ Failed to add channel to folder: {folder_error}")
                # Don't fail the whole operation if folder addition fails
            
            return channel_id
        else:
            logger.error(f"❌ Could not extract channel ID from response: {result}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to create channel '{chat_name}': {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return None

async def update_channel_info(channel_id, new_name, new_photo_message=None, old_name=None):
    """
    Update an existing channel's name and optionally its photo
    
    Args:
        channel_id: The channel ID to update (can be string or int)
        new_name: The new name for the channel
        new_photo_message: Optional message with photo to update channel avatar
        old_name: The old name to compare against (skip title update if same)
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"🔄 Updating channel {channel_id}: name='{new_name}'")
    try:
        from telethon.tl.functions.channels import EditTitleRequest, EditPhotoRequest
        from telethon.tl.types import InputChatUploadedPhoto
        
        # Convert channel_id to int if it's a string, then get the entity
        channel_id_int = int(channel_id) if isinstance(channel_id, str) else channel_id
        channel_entity = await client.get_input_entity(channel_id_int)
        
        title_updated = False
        photo_updated = False
        
        # Only update channel title if it actually changed
        if old_name is None or new_name != old_name:
            await client(EditTitleRequest(
                channel=channel_entity,
                title=new_name
            ))
            logger.info(f"✅ Channel name updated to '{new_name}'")
            title_updated = True
        else:
            logger.info(f"ℹ️ Channel name unchanged ('{new_name}'), skipping title update")
        
        # Update channel photo if provided
        if new_photo_message and new_photo_message.photo:
            import tempfile
            logger.info("🖼️ Updating channel avatar from new image...")
            
            # Download photo to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                photo_path = temp_file.name
                await client.download_media(new_photo_message.photo, photo_path)
                logger.info(f"📥 Photo downloaded to {photo_path}")
            
            # Upload and set as channel photo
            uploaded_file = await client.upload_file(photo_path)
            await client(EditPhotoRequest(
                channel=channel_entity,
                photo=InputChatUploadedPhoto(uploaded_file)
            ))
            
            # Clean up temp file
            os.unlink(photo_path)
            logger.info(f"✅ Channel avatar updated successfully for '{new_name}'")
            photo_updated = True
        
        # Return success if either title or photo was updated
        if title_updated or photo_updated:
            return True
        else:
            logger.info(f"ℹ️ No changes made to channel {channel_id}")
            return True  # Still return True since no error occurred
        
    except Exception as e:
        logger.error(f"❌ Failed to update channel {channel_id}: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False

async def add_channel_to_folder(channel, folder_name):
    """
    Add a channel to a Telegram folder using client.edit_folder()
    """
    logger.info(f"📁 Adding channel to folder: '{folder_name}'")
    try:
        from telethon.tl.functions.messages import GetDialogFiltersRequest
        
        filters_result = await personal_client(GetDialogFiltersRequest())
        filters = filters_result.filters if hasattr(filters_result, 'filters') else []
        
        target_folder_id = None
        for f in filters:
            if hasattr(f, 'title'):
                title_text = f.title.text if hasattr(f.title, 'text') else str(f.title)
                if title_text == folder_name:
                    target_folder_id = f.id
                    logger.info(f"� Found folder '{folder_name}' with ID {target_folder_id}")
                    break
        
        if target_folder_id is None:
            logger.warning(f"⚠️ Folder '{folder_name}' not found - create it in Telegram first")
            return
        
        # Use EditPeerFoldersRequest with InputFolderPeer
        from telethon.tl.functions.messages import UpdateDialogFilterRequest
        from telethon.tl.types import DialogFilter
        
        # Get the full filter object
        target_filter = next((f for f in filters if hasattr(f, 'title') and 
                             (f.title.text if hasattr(f.title, 'text') else str(f.title)) == folder_name), None)
        
        if not target_filter:
            logger.warning(f"⚠️ Could not get filter object")
            return
        
        # Convert channel to proper peer format
        peer = await personal_client.get_input_entity(channel)
        
        # Add to include_peers
        existing_peers = list(target_filter.include_peers) if target_filter.include_peers else []
        existing_peers.append(peer)
        
        await personal_client(UpdateDialogFilterRequest(
            id=target_filter.id,
            filter=DialogFilter(
                id=target_filter.id,
                title=target_filter.title,
                pinned_peers=getattr(target_filter, 'pinned_peers', []),
                include_peers=existing_peers,
                exclude_peers=getattr(target_filter, 'exclude_peers', []),
                emoticon=getattr(target_filter, 'emoticon', None),
                contacts=getattr(target_filter, 'contacts', False),
                non_contacts=getattr(target_filter, 'non_contacts', False),
                groups=getattr(target_filter, 'groups', False),
                broadcasts=getattr(target_filter, 'broadcasts', False),
                bots=getattr(target_filter, 'bots', False),
                exclude_muted=getattr(target_filter, 'exclude_muted', False),
                exclude_read=getattr(target_filter, 'exclude_read', False),
                exclude_archived=getattr(target_filter, 'exclude_archived', False)
            )
        ))
        logger.info(f"✅ Channel added to folder '{folder_name}' successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to add channel to folder: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")

async def send_message(chat_name, thread_id, message):
    return await client.send_message(get_entity_id(chat_name), message, reply_to=thread_id)

async def edit_message(message, new_message):
    try:
        await message.edit(new_message)
        logger.info(f"✏️ Message edited successfully")
    except Exception as e:
        logger.error(f"❌ Error editing message: {e}")
    return message

async def send_video(chat_name, thread_id, video_path, caption=""):
    return await client.send_file(get_entity_id(chat_name), video_path, reply_to=int(thread_id), caption=caption)

# ============================================================================
# SD INPAINTING HELPERS
# ============================================================================
def _ensure_sd_loaded():
    global _sd_pipe, _seg_model
    if _seg_model is not None and _sd_pipe is not None:
        return
    logger.info("📦 Loading SD inpainting models...")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available for PyTorch. Stable Diffusion cloth removal requires a CUDA-capable GPU. "
            "Please start on an environment with an NVIDIA GPU runtime (e.g. Google Colab with GPU enabled)."
        )
    import albumentations as albu
    from diffusers import StableDiffusionInpaintPipeline
    from cloths_segmentation.pre_trained_models import create_model

    _seg_model = create_model('Unet_2020-10-30')
    _seg_model.eval()

    _sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    _sd_pipe.enable_attention_slicing()
    _sd_pipe = _sd_pipe.to('cuda')
    logger.info("✅ SD inpainting models loaded")


def _run_inpaint_sync(image_bytes: bytes) -> bytes:
    """Cloth-removal pipeline. Blocking — always call via run_in_executor."""
    import io, cv2, torch, numpy as np, albumentations as albu
    from PIL import Image
    from iglovikov_helper_functions.utils.image_utils import pad, unpad
    from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

    _ensure_sd_loaded()

    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np  = np.array(pil_image)

    # Segmentation
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded, pads = pad(image_np, factor=32, border=cv2.BORDER_CONSTANT)
    x = torch.unsqueeze(tensor_from_rgb_image(transform(image=padded)['image']), 0)
    with torch.no_grad():
        prediction = _seg_model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    # Dilate to cover all clothing edges
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask,  kernel, iterations=1)

    # Resize to SD-friendly dimensions
    w, h = pil_image.size
    if w > h:
        nw, nh = SD_TARGET_SIZE, int(SD_TARGET_SIZE * h / w)
    else:
        nw, nh = int(SD_TARGET_SIZE * w / h), SD_TARGET_SIZE
    nw, nh = (nw // 8) * 8, (nh // 8) * 8

    img_resized  = pil_image.resize((nw, nh), Image.LANCZOS)
    mask_resized = Image.fromarray(mask * 255, mode='L').resize((nw, nh), Image.NEAREST)

    # Inpaint
    generator = torch.Generator(device='cuda').manual_seed(SD_SEED)
    result = _sd_pipe(
        prompt=SD_PROMPT,
        negative_prompt=SD_NEGATIVE,
        image=img_resized,
        mask_image=mask_resized,
        num_inference_steps=SD_STEPS,
        guidance_scale=SD_GUIDANCE,
        strength=SD_STRENGTH,
        height=nh,
        width=nw,
        generator=generator,
    ).images[0]

    # Blend — preserve non-masked pixels from original
    mask_f  = np.array(mask_resized).astype(np.float32) / 255.0
    mask_3c = np.stack([mask_f] * 3, axis=-1)
    blended = np.array(result) * mask_3c + np.array(img_resized) * (1 - mask_3c)
    final   = Image.fromarray(blended.astype(np.uint8)).resize(pil_image.size, Image.LANCZOS)

    out = io.BytesIO()
    final.save(out, format='JPEG', quality=95)
    return out.getvalue()


async def handle_etcd_image(event):
    """Download image posted in group→etcd thread, inpaint, send result back."""
    logger.info("🔥 ULTRA-AGGRESSIVE CLOTHING REMOVAL MODE ACTIVATED — etcd thread")
    status = await client.send_message(
        get_entity_id('group'),
        '⏳ Processing... ULTRA-AGGRESSIVE CLOTHING REMOVAL MODE ACTIVATED 🔥',
        reply_to=thread_map['etcd'],
    )
    try:
        image_bytes = await client.download_media(event.message, bytes)
        loop = asyncio.get_event_loop()
        result_bytes = await loop.run_in_executor(None, _run_inpaint_sync, image_bytes)

        import io as _io
        buf = _io.BytesIO(result_bytes)
        buf.name = 'result.jpg'
        await client.send_file(
            get_entity_id('group'),
            buf,
            reply_to=thread_map['etcd'],
            caption='✅ Done!',
        )
        await status.delete()
        logger.info("✅ Inpainting result sent to etcd thread")
    except Exception as e:
        logger.error(f"❌ Inpainting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await status.edit(f'❌ Failed: {e}')

# ============================================================================
# MESSAGE HANDLERS
# ============================================================================
async def handle_image_no_text(event):
    await event.reply(
        f"🖼️ **Photo received** (ID: `{event.message.id}`)\n"
        f"💬 Reply to this photo with a name (e.g. `alex`) to register it!"
    )

async def handle_image_has_text(event, custom_message=None, custom_name=None):
    target_msg = custom_message or event.message
    message_id = target_msg.id
    message_text = (custom_name or target_msg.text or "").lower().strip()
    logger.info(f"🖼️ Processing image with text: '{message_text}' (ID: {message_id})")

    # First check if this message_id already exists (this is an edit, not a new message)
    exists, row_idx, old_text, output_chat, topic_id, channel_id, old_name = check_image_by_message_id(message_id)
    
    if exists:
        logger.info(f"✏️ Message ID {message_id} already exists - this is an edit from '{old_text}' to '{message_text}'")
        
        # Update the channel name and photo if channel_id exists
        if channel_id:
            update_success = await update_channel_info(channel_id, message_text, target_msg, old_text)
            if update_success:
                # Update the sheet with new name
                # Column B (index 1) = message_text
                # Column E (index 4) = old_name (keep it the same as message_text for consistency)
                sh.worksheet("list_image").update_acell(f"B{row_idx+2}", message_text)  # Column B = message_text
                sh.worksheet("list_image").update_acell(f"E{row_idx+2}", message_text)  # Column E = old_name
                logger.info(f"✅ Updated sheet entry for message {message_id}")
                
                # Customize message based on what changed
                if old_text != message_text:
                    reply_message = await event.reply(f"✅ Channel '{old_text}' renamed to '{message_text}' and photo updated")
                else:
                    reply_message = await event.reply(f"✅ Channel '{message_text}' photo updated")
            else:
                reply_message = await event.reply(f"⚠️ Failed to update channel '{old_text}'")
        else:
            # Old record without channel - create one now
            logger.info(f"🆕 No channel exists for message {message_id}, creating new channel: {message_text}")
            new_channel_id = await create_separate_chat(message_text, target_msg)
            
            if new_channel_id:
                # Update the sheet with new channel_id in column F (index 5)
                sh.worksheet("list_image").update_acell(f"F{row_idx+2}", new_channel_id)
                # Also update name columns if changed
                if old_text != message_text:
                    sh.worksheet("list_image").update_acell(f"B{row_idx+2}", message_text)
                    sh.worksheet("list_image").update_acell(f"E{row_idx+2}", message_text)
                logger.info(f"✅ Created new channel with ID: {new_channel_id}")
                reply_message = await event.reply(f"✅ Channel created for '{message_text}'")
            else:
                logger.error(f"❌ Failed to create channel for message {message_id}")
                reply_message = await event.reply(f"⚠️ Failed to create channel for '{message_text}'")
        
        sleep(2)
        await reply_message.delete()
        return

    # Not an edit - check if topic with this name already exists (different message)
    exist_topic = check_topic_exist(message_text)

    if exist_topic[0]:
        logger.info(f"📂 Topic already exists: {exist_topic[1]}")
        reply_message = await event.reply(f"Topic already exist with name {exist_topic[1]}")
        update_topic_id(message_id, message_text)
        await reply_message.edit(f"Topic already exist with name {exist_topic[1]} and updated the id")
        sleep(2)
        await reply_message.delete()
        return

    # New message with new name - create everything
    logger.info(f"🆕 Creating forum topic and separate channel: {message_text}")
    
    # Run both operations in parallel using asyncio.gather
    topic_task = create_forum_topic(message_text, get_entity_id("output_chat_id"))
    channel_task = create_separate_chat(message_text, target_msg)  # Pass the message with photo
    
    # Wait for both to complete
    topic_id, channel_id = await asyncio.gather(topic_task, channel_task)

    if topic_id is not None:
        # Old structure: [message_id, message_text, output_chat_id, topic_id, message_text]
        # New structure: [message_id, message_text, output_chat_id, topic_id, message_text, channel_id]
        # Append channel_id at the END to not break old records
        sh.worksheet("list_image").append_row([message_id, message_text, get_entity_id("output_chat_id"), topic_id, message_text, channel_id or ""])
        logger.info(f"✅ Topic '{message_text}' created successfully with ID: {topic_id}")
        
        if channel_id is not None:
            logger.info(f"✅ Separate channel '{message_text}' created with ID: {channel_id}")
            reply_message = await event.reply(f"✅ Topic and channel created for {message_text}\n📂 Topic ID: {topic_id}\n📢 Channel ID: {channel_id}")
        else:
            logger.warning(f"⚠️ Channel creation failed but topic succeeded")
            reply_message = await event.reply(f"⚠️ Topic {message_text} created (channel failed)")
        
        sleep(2)
        await reply_message.delete()
    else:
        logger.error(f"❌ Failed to create topic: {message_text}")
        if channel_id is not None:
            logger.info(f"⚠️ Channel created but topic failed for {message_text}")

    logger.info(f"✅ Completed handling image with text: {message_text}")

async def register_or_update_video(message_id, video_name, event):
    message_id = str(message_id)
    video_name = video_name.strip().lower()

    if not video_name:
        await event.reply("⚠️ Video name cannot be empty.")
        return False

    exists, old_name, row_num = check_video_exist(message_id)
    if exists:
        if old_name == video_name:
            reply_msg = await event.reply(f"ℹ️ Video already registered with name '{video_name}' (ID: {message_id})")
            sleep(3)
            await reply_msg.delete()
            return True
        try:
            sh.worksheet("list_video").update_acell(f"B{row_num}", video_name)
            logger.info(f"✏️ Updated video name for ID {message_id}: '{old_name}' → '{video_name}'")
            reply_msg = await event.reply(f"✏️ Video (ID: {message_id}) renamed from '{old_name}' to '{video_name}'")
            sleep(3)
            await reply_msg.delete()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to rename video {message_id}: {e}")
            await event.reply(f"❌ Error updating video name: {e}")
            return False
    else:
        try:
            sh.worksheet("list_video").append_row([message_id, video_name])
            logger.info(f"✅ Video '{video_name}' (ID: {message_id}) added successfully")
            reply_msg = await event.reply(f"✅ Video '{video_name}' added successfully (ID: {message_id})")
            sleep(3)
            await reply_msg.delete()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add video {message_id}: {e}")
            await event.reply(f"❌ Error adding video: {e}")
            return False

async def handle_input_video_no_text(event):
    await event.reply(
        f"📹 **Video received** (ID: `{event.message.id}`)\n"
        f"💬 Reply to this video with a name (e.g. `dance1`) to save it!"
    )

async def handle_input_video_has_text(event):
    message_id = event.message.id
    raw_text = (event.message.text or "").strip()
    clean_name = sanitize_video_name(raw_text)
    if clean_name:
        await register_or_update_video(message_id, clean_name, event)
    else:
        await handle_input_video_no_text(event)

async def handle_message_deleted(event):
    """Handle when messages are deleted from input_chat_id"""
    logger.info(f"🗑️ Message deletion detected")

    deleted_count = 0
    for message_id in event.deleted_ids:
        logger.info(f"🔍 Checking deleted message ID: {message_id}")

        success, image_name = delete_image_by_message_id(message_id)
        if success:
            deleted_count += 1
            output_count, deleted_outputs = delete_outputs_by_image_id(message_id)
            if output_count > 0:
                logger.info(f"🧹 Cleaned up {output_count} related output(s) for deleted image '{image_name}'")
            continue

        success, video_name = delete_video_by_message_id(message_id)
        if success:
            deleted_count += 1
            continue

        logger.info(f"ℹ️ Message ID {message_id} not found in any sheet (might be untracked message)")

    if deleted_count > 0:
        logger.info(f"✅ Successfully processed {deleted_count} deleted message(s)")
    else:
        logger.info(f"ℹ️ No matching entries found in sheets for deleted messages")

async def domany(event):
    message_parts = event.message.text.strip().split()
    if len(message_parts) < 3:
        await event.reply("⚠️ Usage: `/domany <images|all> <videos|all>`")
        return

    image_map = create_map_user_image()
    video_map = create_map_video()

    raw_images = message_parts[1].strip()
    raw_videos = message_parts[2].strip()

    # Expand 'all' or comma-separated list
    if raw_images.lower() == "all":
        image_names = list(image_map.keys())
    else:
        image_names = [x.strip().lower() for x in raw_images.split(",") if x.strip()]

    if raw_videos.lower() == "all":
        video_names = list(video_map.keys())
    else:
        video_names = [x.strip().lower() for x in raw_videos.split(",") if x.strip()]

    if not image_names or not video_names:
        await event.reply("⚠️ No images or videos found to process.")
        return

    # Check for non-existent items
    missing_images = [img for img in image_names if img not in image_map]
    missing_videos = [vid for vid in video_names if vid not in video_map]
    if missing_images or missing_videos:
        error_msgs = []
        if missing_images:
            error_msgs.append(f"📸 Missing images: {', '.join(missing_images)}")
        if missing_videos:
            error_msgs.append(f"🎥 Missing videos: {', '.join(missing_videos)}")
        await event.reply(f"❌ Cannot proceed:\n" + "\n".join(error_msgs))
        return

    total = len(image_names) * len(video_names)

    logger.info(f"🚀 Starting batch processing:")
    logger.info(f"   📸 Images ({len(image_names)}): {', '.join(image_names)}")
    logger.info(f"   🎥 Videos ({len(video_names)}): {', '.join(video_names)}")
    logger.info(f"   📊 Total combinations: {total}")

    progress = 0
    success = 0
    skip = 0
    queued = 0
    fail_list = []

    tracking_message = await send_message("group", thread_map["output"], 
                                         f"🎬 Starting batch render: {len(image_names)} images × {len(video_names)} videos = {total} total combinations")
    current_data = get_data("list_output")

    # Callbacks for tracking updates from queue worker
    async def success_callback(combo_name):
        nonlocal success
        success += 1
        tracking_content = f"📊 Progress: {progress}/{total}\n✅ Success: {success} | ⚡ Skip: {skip} | 🔄 Queued: {queued} | ❌ Fail: {len(fail_list)}"
        if fail_list:
            tracking_content += f"\n\n❌ Failed items:\n{chr(10).join(fail_list)}"
        await edit_message(tracking_message, tracking_content)

    async def failure_callback(combo_name, error_msg):
        fail_list.append(combo_name)
        tracking_content = f"📊 Progress: {progress}/{total}\n✅ Success: {success} | ⚡ Skip: {skip} | 🔄 Queued: {queued} | ❌ Fail: {len(fail_list)}"
        if fail_list:
            tracking_content += f"\n\n❌ Failed items:\n{chr(10).join(fail_list)}"
        await edit_message(tracking_message, tracking_content)

    for image_name in image_names:
        for video_name in video_names:
            progress += 1
            combo_name = f"{image_name}_{video_name}"
            logger.info(f"🔄 Processing [{progress}/{total}]: {combo_name}")

            image_id = image_map[image_name].get("message_id")
            video_id = video_map[video_name].get("message_id")
            channel_id = image_map[image_name].get("channel_id")
            output_exist = check_output_exist(f"{image_id}_{video_id}", current_data)

            try:
                if output_exist[0]:
                    logger.info(f"⚡ Output {combo_name} already exists, skipping")
                    skip += 1
                    continue

                tracking_message_content = f"📊 Progress: {progress}/{total}\n✅ Success: {success} | ⚡ Skip: {skip} | 🔄 Queued: {queued} | ❌ Fail: {len(fail_list)}\n🔄 Current: {combo_name}"
                await edit_message(tracking_message, tracking_message_content)

                logger.info(f"📥 Downloading files for {combo_name}")
                input_image = await download_file(message_id=image_id, sub_path="./image", ext="jpg")
                input_video = await download_file(message_id=video_id, sub_path="./video", ext="mp4")
                output_path = f"{os_dir['output']}/{image_id}_{video_id}.mp4"

                if not os.path.exists(input_image) or not os.path.exists(input_video):
                    raise Exception("Input files not found after download")

                if os.path.exists(output_path):
                    logger.info(f"📤 Output file exists, sending to Telegram: {combo_name}")
                    output_thread = get_topic_id(image_id)
                    
                    # Send to forum topic
                    await send_video("output_chat_id", output_thread, output_path, combo_name)
                    logger.info(f"✅ Video sent to forum topic successfully: {combo_name}")
                    
                    # Send to channel if channel_id exists
                    if channel_id:
                        await client.send_file(int(channel_id), output_path, caption=combo_name)
                        logger.info(f"✅ Video sent to channel successfully: {combo_name}")
                    
                    await send_message("input_chat_id", thread_map["etcd"], f"✅ {combo_name} completed successfully")
                    sh.worksheet("list_output").append_row([f"{image_id}_{video_id}", combo_name])
                    success += 1
                else:
                    # Queue the rendering task instead of blocking
                    logger.info(f"🔄 Queueing video rendering: {combo_name}")
                    output_thread = get_topic_id(image_id)
                    
                    task_data = {
                        'combo_name': combo_name,
                        'input_image': input_image,
                        'input_video': input_video,
                        'output_path': output_path,
                        'output_thread': output_thread,
                        'channel_id': channel_id,
                        'image_id': image_id,
                        'video_id': video_id,
                        'tracking_message': tracking_message,
                        'success_callback': success_callback,
                        'failure_callback': failure_callback
                    }
                    
                    await render_queue.put(task_data)
                    queued += 1
                    logger.info(f"✅ {combo_name} queued for rendering (queue size: {render_queue.qsize()})")

            except Exception as e:
                logger.error(f"❌ FAILED processing {combo_name}: {str(e)}")
                await send_message("group", thread_map["etcd"], f"❌ {combo_name} failed: {str(e)}")
                fail_list.append(combo_name)

            finally:
                tracking_message_content = f"📊 Progress: {progress}/{total}\n✅ Success: {success} | ⚡ Skip: {skip} | 🔄 Queued: {queued} | ❌ Fail: {len(fail_list)}"
                if fail_list:
                    tracking_message_content += f"\n\n❌ Failed items:\n{chr(10).join(fail_list)}"
                await edit_message(tracking_message, tracking_message_content)

    logger.info(f"🏁 Batch processing completed!")
    logger.info(f"   📊 Total processed: {progress}/{total}")
    logger.info(f"   ✅ Successful: {success}")
    logger.info(f"   ⚡ Skipped: {skip}")
    logger.info(f"   🔄 Queued: {queued}")
    logger.info(f"   ❌ Failed: {len(fail_list)}")
    if fail_list:
        logger.info(f"   📝 Failed items: {', '.join(fail_list)}")
    
    # Notify user that tasks are queued
    if queued > 0:
        await event.reply(f"✅ Batch queued: {queued} videos will render in background. Bot remains responsive!")

def is_topic_reply(topic, event):
    if not event.reply_to:
        return False
    target = entity_map["input_chat_id"]["threads"].get(topic)
    if target is None:
        return False
    reply_to_msg_id = getattr(event.reply_to, 'reply_to_msg_id', None)
    reply_to_top_id = getattr(event.reply_to, 'reply_to_top_id', None)
    return reply_to_msg_id == target or reply_to_top_id == target

# ============================================================================
# EVENT HANDLERS
# ============================================================================
@client.on(events.NewMessage(chats=get_entity_id("input_chat_id"), from_users=get_entity_id("user_id")))
async def new_photo_handler(event):
    logger.info(f"📨 New message received from user {get_entity_id('user_id')}")

    if is_topic_reply("faces", event):
        if is_photo_message(event.message):
            logger.info("🖼️ Processing photo in faces topic")
            if event.message.text:
                await handle_image_has_text(event)
            else:
                await handle_image_no_text(event)
            return

        elif event.message.text:
            text = event.message.text.strip()
            if text.startswith("/"):
                return
            logger.info(f"💬 Received text in faces topic: '{text}' (resolving target photo)")
            target_photo = None

            if event.reply_to and event.reply_to.reply_to_msg_id:
                reply_id = event.reply_to.reply_to_msg_id
                topic_root = entity_map["input_chat_id"]["threads"]["faces"]
                if reply_id != topic_root:
                    target_msg = await client.get_messages(get_entity_id("input_chat_id"), ids=reply_id)
                    if is_photo_message(target_msg):
                        target_photo = target_msg
                    elif target_msg and target_msg.reply_to and target_msg.reply_to.reply_to_msg_id:
                        parent_msg = await client.get_messages(get_entity_id("input_chat_id"), ids=target_msg.reply_to.reply_to_msg_id)
                        if is_photo_message(parent_msg):
                            target_photo = parent_msg

            if not target_photo:
                recent_msgs = await client.get_messages(
                    get_entity_id("input_chat_id"),
                    limit=10,
                    reply_to=entity_map["input_chat_id"]["threads"]["faces"]
                )
                for m in recent_msgs:
                    if m.id != event.message.id and is_photo_message(m):
                        exists, _, _, _, _, _, _ = check_image_by_message_id(m.id)
                        if not exists:
                            target_photo = m
                            logger.info(f"🎯 Matched text '{text}' to recent unnamed photo ID {m.id}")
                            break

            if target_photo:
                await handle_image_has_text(event, custom_message=target_photo, custom_name=text.lower())
                return
            else:
                logger.info("❌ Message in faces topic but no photo found")
                return

    if is_topic_reply("vid", event):
        if is_video_message(event.message):
            logger.info(f"🎥 Processing video in vid topic (ID: {event.message.id})")
            if event.message.text:
                await handle_input_video_has_text(event)
            else:
                await handle_input_video_no_text(event)
            return

        elif event.message.text:
            text = event.message.text.strip()
            if text.startswith("/"):
                return
            logger.info(f"💬 Received text in vid topic: '{text}' (resolving target video)")
            target_video_id = None

            # 1. Check direct reply
            if event.reply_to and event.reply_to.reply_to_msg_id:
                reply_id = event.reply_to.reply_to_msg_id
                topic_root = entity_map["input_chat_id"]["threads"]["vid"]
                if reply_id != topic_root:
                    target_msg = await client.get_messages(get_entity_id("input_chat_id"), ids=reply_id)
                    if is_video_message(target_msg):
                        target_video_id = target_msg.id
                    elif target_msg and target_msg.reply_to and target_msg.reply_to.reply_to_msg_id:
                        parent_msg = await client.get_messages(get_entity_id("input_chat_id"), ids=target_msg.reply_to.reply_to_msg_id)
                        if is_video_message(parent_msg):
                            target_video_id = parent_msg.id

            # 2. Check recent unnamed video
            if not target_video_id:
                recent_msgs = await client.get_messages(
                    get_entity_id("input_chat_id"),
                    limit=10,
                    reply_to=entity_map["input_chat_id"]["threads"]["vid"]
                )
                for m in recent_msgs:
                    if m.id != event.message.id and is_video_message(m):
                        exists, _, _ = check_video_exist(m.id)
                        if not exists:
                            target_video_id = m.id
                            logger.info(f"🎯 Matched text '{text}' to recent unnamed video ID {target_video_id}")
                            break

            if target_video_id:
                clean_name = sanitize_video_name(text)
                await register_or_update_video(target_video_id, clean_name, event)
                return
            else:
                logger.info(f"ℹ️ Text in vid topic not linked to any video: '{text}'")
                await event.reply("⚠️ Please reply directly to a video message with the name you want to give it.")
                return

        else:
            logger.info("❌ Message in vid topic but no video or text found")
            return

@client.on(events.MessageEdited(chats=get_entity_id("input_chat_id"), from_users=get_entity_id("user_id")))
async def message_edited_handler(event):
    logger.info(f"✏️ Message edited by user {get_entity_id('user_id')}")

    if is_topic_reply("faces", event):
        if is_photo_message(event.message) and event.message.text:
            logger.info("🖼️ Processing edited photo with text in faces topic")
            await handle_image_has_text(event)
        return

    if is_topic_reply("vid", event):
        if is_video_message(event.message) and event.message.text:
            logger.info("🎥 Processing edited video with text in vid topic")
            clean_name = sanitize_video_name(event.message.text)
            if clean_name:
                await register_or_update_video(event.message.id, clean_name, event)
        return

@client.on(events.NewMessage(chats=get_entity_id("group"), from_users=get_entity_id("user_id"), pattern='/domany .*'))
async def domany_handler(event):
    logger.info(f"🎯 Received /domany command from user")
    await domany(event)

@client.on(events.NewMessage(chats=get_entity_id("group"), pattern='/get_chat_id'))
async def get_chat_id(event):
    logger.info(f"🆔 Chat ID requested")
    logger.info(f"Event details: {event.stringify()}")
    await event.reply(f"💬 Chat ID: {event.chat_id}")

@client.on(events.NewMessage(chats=get_entity_id("group"), pattern='/getres'))
async def get_resource(event):
    logger.info(f"📋 Resource list requested")
    image_list = create_map_user_image()
    video_list = create_map_video()
    image_list_key = ",".join(image_list.keys())
    video_list_key = ",".join(video_list.keys())
    logger.info(f"📸 Available images: {len(image_list)} items")
    logger.info(f"🎥 Available videos: {len(video_list)} items")
    await event.reply(f"📋 **Available Resources:**\n\n📸 **Images:** {image_list_key}\n\n🎥 **Videos:** {video_list_key}")

@client.on(MessageDeleted(chats=get_entity_id("input_chat_id")))
async def message_deleted_handler(event):
    logger.info(f"🗑️ Message deletion event detected in input_chat_id")
    await handle_message_deleted(event)

@client.on(events.NewMessage(chats=get_entity_id("group"), from_users=get_entity_id("user_id"), pattern='/delete_image .*'))
async def delete_image_command(event):
    """Handle manual image deletion by name"""
    try:
        command_parts = event.message.text.split(" ", 1)
        if len(command_parts) != 2:
            await event.reply("❌ Usage: /delete_image <image_name>")
            return

        image_name = command_parts[1].lower().strip()
        logger.info(f"🗑️ Manual image deletion requested: {image_name}")

        data = get_data("list_image")
        found = False
        for i in range(len(data)):
            if data[i][1] == image_name:
                message_id = data[i][0]
                row_number = i + 2
                sh.worksheet("list_image").delete_rows(row_number)

                output_count, deleted_outputs = delete_outputs_by_image_id(message_id)

                success_msg = f"✅ Deleted image '{image_name}' (ID: {message_id})"
                if output_count > 0:
                    success_msg += f"\n🧹 Also deleted {output_count} related output(s)"

                await event.reply(success_msg)
                logger.info(f"✅ Successfully deleted image '{image_name}' manually")
                found = True
                break

        if not found:
            await event.reply(f"❌ Image '{image_name}' not found")
            logger.warning(f"⚠️ Manual deletion failed: image '{image_name}' not found")

    except Exception as e:
        logger.error(f"❌ Error in manual image deletion: {e}")
        await event.reply(f"❌ Error deleting image: {str(e)}")

@client.on(events.NewMessage(chats=get_entity_id("group"), from_users=get_entity_id("user_id"), pattern='/delete_video .*'))
async def delete_video_command(event):
    """Handle manual video deletion by name"""
    try:
        command_parts = event.message.text.split(" ", 1)
        if len(command_parts) != 2:
            await event.reply("❌ Usage: /delete_video <video_name>")
            return

        video_name = command_parts[1].lower().strip()
        logger.info(f"🗑️ Manual video deletion requested: {video_name}")

        data = get_data("list_video")
        found = False
        for i in range(len(data)):
            if data[i][1] == video_name:
                message_id = data[i][0]
                row_number = i + 2
                sh.worksheet("list_video").delete_rows(row_number)

                await event.reply(f"✅ Deleted video '{video_name}' (ID: {message_id})")
                logger.info(f"✅ Successfully deleted video '{video_name}' manually")
                found = True
                break

        if not found:
            await event.reply(f"❌ Video '{video_name}' not found")
            logger.warning(f"⚠️ Manual deletion failed: video '{video_name}' not found")

    except Exception as e:
        logger.error(f"❌ Error in manual video deletion: {e}")
        await event.reply(f"❌ Error deleting video: {str(e)}")

@client.on(events.NewMessage(chats=get_entity_id('group'), from_users=get_entity_id('user_id')))
async def etcd_image_handler(event):
    """Trigger SD inpainting when user posts a photo in the group etcd thread (375)."""
    if not event.photo:
        return
    if not event.reply_to or event.reply_to.reply_to_msg_id != thread_map['etcd']:
        return
    logger.info(f"🖼️ Photo received in group etcd thread — starting inpainting (msg {event.message.id})")
    await handle_etcd_image(event)

@client.on(events.NewMessage(chats=get_entity_id("group"), pattern='/check_cuda'))
async def check_cuda_command(event):
    """Handle /check_cuda command to report hardware and acceleration status to Telegram"""
    logger.info("🔍 /check_cuda command received")
    status = check_cuda_and_prerequisites(silent=True)
    res = status['results']
    v_st = "✅ CUDA Accelerated" if status['video_cuda'] else "⚠️ CPU Fallback (Slow)"
    i_st = "✅ CUDA Accelerated" if status['image_cuda'] else "❌ CUDA Unavailable"
    
    msg = (
        f"🖥️ **System & Acceleration Status:**\n\n"
        f"• **NVIDIA GPU:** `{res.get('nvidia_gpu', ('?', '?'))[1]}`\n"
        f"• **PyTorch CUDA:** `{res.get('torch_cuda', ('?', '?'))[1]}`\n"
        f"• **ONNX Runtime:** `{res.get('ort_cuda', ('?', '?'))[1]}`\n"
        f"• **FFmpeg:** `{res.get('ffmpeg', ('?', '?'))[1]}`\n\n"
        f"📊 **Readiness:**\n"
        f"• 🎥 Video Face Swap: {v_st}\n"
        f"• 🖼️ Image Inpainting: {i_st}\n"
    )
    if status['issues']:
        msg += "\n⚠️ **Issues Detected:**\n"
        for idx, (comp, reason, fix) in enumerate(status['issues'], 1):
            msg += f"{idx}. **{comp}**: {reason}\n   👉 _Fix_: `{fix}`\n"
    await event.reply(msg)

# ============================================================================
# MAIN FUNCTION
# ============================================================================
async def main():
    """Main function to run the Telegram bot"""
    global render_worker_tasks

    # Step 1: Pre-flight Hardware & CUDA Prerequisite Check
    check_cuda_and_prerequisites()

    log_separator("STARTING TELEGRAM CLIENT")
    logger.info("🚀 Starting Telegram clients...")

    import json
    roop_config_file = os.getenv('ROOP_CONFIG_FILE', './roop_config.json')
    concurrent_renders = 1
    if os.path.exists(roop_config_file):
        with open(roop_config_file) as f:
            concurrent_renders = json.load(f).get('concurrent_renders', 1)

    async with client, personal_client:
        await client.start()
        await personal_client.start()
        logger.info("✅ Both clients started successfully")

        # Start N render queue workers (one per concurrent render slot)
        render_worker_tasks = [
            asyncio.create_task(process_render_queue())
            for _ in range(concurrent_renders)
        ]
        logger.info(f"🎬 Started {concurrent_renders} render worker(s)")
        
        logger.info("👂 Bot is now listening for events...")
        logger.info("🔧 Available commands:")
        logger.info("   • /domany <images|all> <videos|all> - Start batch processing (non-blocking)")
        logger.info("   • /get_chat_id - Get current chat ID")
        logger.info("   • /getres - List available resources")
        logger.info("   • /check_cuda - Check GPU & CUDA acceleration status")
        logger.info("   • /delete_image <name> - Manually delete image by name")
        logger.info("   • /delete_video <name> - Manually delete video by name")
        logger.info("🔥 SD inpainting: upload a photo to group etcd thread (375) to trigger cloth removal")
        logger.info("🗑️ Auto-deletion enabled: Deleting messages will remove entries from sheets")
        logger.info("⚡ Non-blocking rendering: Bot stays responsive while processing videos")
        log_separator()
        print("Start Polling")

        try:
            await asyncio.sleep(6000000)
        except asyncio.CancelledError:
            logger.info("⚠️ Received cancellation signal")
        except KeyboardInterrupt:
            logger.info("⚠️ Received keyboard interrupt")
        finally:
            logger.info("🛑 Stopping bot...")
            
            # Send one shutdown signal per worker, then await all
            for _ in render_worker_tasks:
                await render_queue.put(None)
            for task in render_worker_tasks:
                await task
            
            print("Stopping")
            await client.disconnect()
            logger.info("✅ Bot stopped successfully")
            print("Stopped")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--check", "--check-cuda", "-c", "--prereq"):
        check_cuda_and_prerequisites()
        sys.exit(0)
    asyncio.run(main())
