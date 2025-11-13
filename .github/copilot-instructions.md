# Roop - AI Face Swap Tool

## Architecture Overview

This is a face-swapping application using deep learning models (InsightFace, GFPGAN) with ONNX Runtime for inference. The codebase follows a **functional programming paradigm exclusively** - no OOP patterns are accepted.

### Core Processing Pipeline

1. **Entry Point**: `run.py` → `roop/core.py:run()`
2. **Dual Modes**: GUI mode (customtkinter) or CLI mode (when `-s/--source` provided)
3. **Video Processing Flow**:
   - Extract frames to `temp/<video_name>/*.png` using ffmpeg
   - Process each frame with frame processors (pipeline pattern)
   - Reassemble video with original/custom fps and audio
   - Clean up temp files (unless `--keep-frames`)

### Frame Processor Plugin System

Located in `roop/processors/frame/`, each processor must implement:
- `pre_check()` - Download models, verify dependencies
- `pre_start()` - Validate inputs before processing
- `process_frame(source_face, temp_frame)` - Single frame logic
- `process_image(source_path, target_path, output_path)` - Image workflow
- `process_video(source_path, temp_frame_paths)` - Video workflow

**Current Processors**:
- `face_swapper.py` - Uses InsightFace inswapper_128.onnx model
- `face_enhancer.py` - Uses GFPGAN for upscaling/enhancement

Models auto-download from HuggingFace on first run (~300MB).

## Key Conventions

### Global State Management
All runtime config lives in `roop/globals.py` as module-level variables (not a class). Updated from CLI args in `core.py:parse_args()`.

### Thread Safety Pattern
Processors use double-lock for lazy model initialization:
```python
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

def get_face_swapper():
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            FACE_SWAPPER = insightface.model_zoo.get_model(...)
    return FACE_SWAPPER
```

### Execution Providers
Platform-specific ONNX acceleration configured via `--execution-provider`:
- macOS: `coreml` (M1/M2) or `cpu`
- NVIDIA: `cuda` 
- AMD: `rocm`
- Windows DirectML: `dml`

**Critical**: Set `OMP_NUM_THREADS=1` when using GPU providers (doubles CUDA performance).

### NSFW Content Filtering
Built-in safety via `opennsfw2` in `predicter.py`:
- Images/videos over 85% probability threshold are rejected
- Videos sample every 100 frames
- Cannot be disabled (ethical requirement)

## Development Workflows

### Running the Application
```bash
# GUI mode
python run.py

# CLI mode (image)
python run.py -s source.jpg -t target.jpg -o output.jpg

# CLI mode (video with GPU)
python run.py -s face.jpg -t video.mp4 -o result.mp4 \
  --execution-provider cuda --many-faces --keep-fps
```

### Dependencies
- **Platform-specific PyTorch**: Check `requirements.txt` for `sys_platform` conditionals
- **ONNX Runtime**: Different packages for macOS (Intel/Silicon) vs GPU systems
- **ffmpeg**: Must be in PATH (checked in `pre_check()`)

### Google Colab Setup
Colab requires special handling due to pre-installed packages:
- **DO NOT** use standard `requirements.txt` - causes dependency conflicts
- Use `requirements-colab.txt` or `setup_colab.py` instead
- Key conflicts: NumPy (Colab needs >=2.0), Torch versions, CUDA versions
- See `COLAB_SETUP.md` for complete installation guide
- Skip torch/torchvision installation - use Colab's pre-installed versions

### Adding a New Frame Processor
1. Create `roop/processors/frame/my_processor.py`
2. Implement all 5 required interface methods
3. Add choice to `--frame-processor` argparse options in `core.py`
4. Download models in `pre_check()` using `conditional_download()`
5. Use `THREAD_LOCK` for thread-safe model initialization

## Common Patterns

### Path Resolution
Use `utilities.resolve_relative_path()` for model paths - resolves relative to `roop/` module directory.

### Video Operations
All video ops use ffmpeg through `utilities.run_ffmpeg()`:
- Hardware acceleration enabled by default (`-hwaccel auto`)
- Frame extraction uses 16 threads, rgb24 pixel format
- Video encoding respects `--video-encoder` (libx264/libx265/libvpx-vp9)

### Face Detection
`face_analyser.py` uses InsightFace buffalo_l model:
- `get_one_face()` returns leftmost face (min bbox x-coord)
- `get_many_faces()` processes all detected faces (requires `--many-faces`)

### Temporary Files
Video processing creates `temp/<video_name>/` in target directory:
- `%04d.png` for frames (0001.png, 0002.png...)
- `temp.mp4` for intermediate output
- Auto-cleanup unless `--keep-frames` flag

## Code Style Requirements (CONTRIBUTING.md)

- **Functional only** - absolutely no OOP/classes
- Self-documenting names - no comments explaining code
- One feature per PR, no fundamental architecture changes
- Testing required before submission
- Resolve CI failures before merging

## Platform-Specific Quirks

- **macOS**: SSL monkey-patched for model downloads (`ssl._create_unverified_context`)
- **macOS memory**: Units in different scale (GB → `* 1024^6` not `1024^3`)
- **Windows**: Uses `ctypes.windll.kernel32` for memory limits
- **Linux/macOS**: Uses `resource.setrlimit()` for memory limits
