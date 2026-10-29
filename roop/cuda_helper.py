import os
import sys
import glob
import ctypes

_ort_repair_attempted = False


def has_nvidia_gpu() -> bool:
    """Return True if an NVIDIA GPU device or nvidia-smi is present on this Linux machine."""
    if sys.platform != 'linux':
        return False
    if os.path.exists('/dev/nvidia0') or os.path.exists('/proc/driver/nvidia'):
        return True
    import shutil
    return shutil.which('nvidia-smi') is not None


def ensure_ort_gpu_installed(force: bool = False) -> bool:
    """
    On Linux with an NVIDIA GPU, ensures that `onnxruntime-gpu` is cleanly installed
    and has NOT been overwritten by the CPU-only `onnxruntime` package (which packages
    like `insightface` pull in as a dependency during `pip install -r requirements.txt`).

    Must ideally be called BEFORE `import onnxruntime` so the current process loads
    the GPU C++ extension (`onnxruntime_pybind11_state.so`).
    Returns True if a repair was performed.
    """
    global _ort_repair_attempted
    if _ort_repair_attempted and not force:
        return False
    if not has_nvidia_gpu():
        return False

    import importlib
    import importlib.metadata
    import importlib.util
    import subprocess

    has_cpu_pkg = False
    try:
        importlib.metadata.distribution('onnxruntime')
        has_cpu_pkg = True
    except importlib.metadata.PackageNotFoundError:
        pass

    has_gpu_pkg = False
    try:
        importlib.metadata.distribution('onnxruntime-gpu')
        has_gpu_pkg = True
    except importlib.metadata.PackageNotFoundError:
        pass

    has_cuda_so = False
    try:
        spec = importlib.util.find_spec('onnxruntime')
        if spec and spec.submodule_search_locations:
            ort_dir = spec.submodule_search_locations[0]
            has_cuda_so = bool(glob.glob(os.path.join(ort_dir, 'capi', '*cuda*.so')))
    except Exception:
        pass

    needs_repair = force or has_cpu_pkg or (not has_gpu_pkg) or (not has_cuda_so)
    if not needs_repair:
        return False

    _ort_repair_attempted = True
    reason = (
        "conflicting CPU 'onnxruntime' package detected alongside 'onnxruntime-gpu'"
        if has_cpu_pkg else
        "missing 'onnxruntime-gpu' CUDA provider libraries"
    )
    print(f"🔧 [CUDA Auto-Repair] {reason} — reinstalling 'onnxruntime-gpu'...", flush=True)
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu'],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        res = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'onnxruntime-gpu'],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print("✅ [CUDA Auto-Repair] 'onnxruntime-gpu' cleanly installed!", flush=True)
            importlib.invalidate_caches()
            return True
        else:
            print(
                f"⚠️ [CUDA Auto-Repair] pip install onnxruntime-gpu returned {res.returncode}: "
                f"{res.stderr[-300:]}",
                flush=True,
            )
    except Exception as e:
        print(f"⚠️ [CUDA Auto-Repair] Failed to auto-repair onnxruntime-gpu: {e}", flush=True)
    return False


def preload_cuda_libraries():
    """
    Preloads NVIDIA CUDA runtime, cuBLAS, and cuDNN shared libraries into the process global symbol table.
    On Linux / Google Colab environments, libraries like libcudart.so, libcublasLt.so, and libcudnn.so
    are frequently installed via pip into nvidia/*/lib or located in /usr/local/cuda*/lib64, but are not
    in the default ld.so search path. Preloading them with RTLD_GLOBAL allows onnxruntime's CUDA provider
    to link against them dynamically without failing.
    """
    # Auto-repair CPU vs GPU onnxruntime package collision before onnxruntime is imported
    if 'onnxruntime' not in sys.modules:
        ensure_ort_gpu_installed()

    search_dirs = [
        '/usr/local/cuda/targets/x86_64-linux/lib',
        '/usr/local/cuda-13/targets/x86_64-linux/lib',
        '/usr/local/cuda-13.0/targets/x86_64-linux/lib',
        '/usr/local/cuda-12/targets/x86_64-linux/lib',
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-13/lib64',
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda-12/lib64',
        '/usr/local/cuda/lib',
        '/usr/lib64-nvidia',
    ]

    # Discover pip-installed nvidia packages in Python site-packages
    for p in sys.path:
        if not p or not os.path.isdir(p):
            continue
        nvidia_root = os.path.join(p, 'nvidia')
        if os.path.isdir(nvidia_root):
            for sub in os.listdir(nvidia_root):
                lib_sub = os.path.join(nvidia_root, sub, 'lib')
                if os.path.isdir(lib_sub):
                    search_dirs.append(lib_sub)
        # Check torch/lib as well
        torch_lib = os.path.join(p, 'torch', 'lib')
        if os.path.isdir(torch_lib):
            search_dirs.append(torch_lib)

    # Update LD_LIBRARY_PATH environment variable for child subprocesses
    cur_ld = os.environ.get('LD_LIBRARY_PATH', '')
    ld_paths = cur_ld.split(':') if cur_ld else []
    for d in search_dirs:
        if os.path.isdir(d) and d not in ld_paths:
            ld_paths.insert(0, d)
    os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_paths)

    # Preload all CUDA and cuDNN 9 modules (ops, cnn, adv, engines, etc.)
    # In cuDNN 9, libcudnn.so dynamically depends on all submodules
    patterns = [
        'libcudart.so*',
        'libcublasLt.so*',
        'libcublas.so*',
        'libcudnn*.so*',
        'libcufft.so*',
        'libcurand.so*',
        'libnvrtc.so*',
    ]

    loaded = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            for so_file in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(so_file))
                except Exception:
                    pass

    return loaded


def inspect_ort_cuda():
    """
    Tests whether onnxruntime can use CUDAExecutionProvider.
    If not available, attempts to load libonnxruntime_providers_cuda.so directly
    using ctypes to catch the exact dynamic linker error (e.g. missing .so file).
    Returns (status: bool, message: str, exact_dl_error: str)
    """
    # First ensure libraries are preloaded
    preload_cuda_libraries()

    try:
        import onnxruntime as ort
    except Exception as e:
        return False, f"Failed to import onnxruntime: {e}", str(e)

    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        return True, f"CUDAExecutionProvider available (ORT v{ort.__version__})", ""

    # Not in providers - find libonnxruntime_providers_cuda.so and diagnose why
    dl_error = ""
    try:
        ort_dir = os.path.dirname(ort.__file__)
        cuda_so_list = glob.glob(os.path.join(ort_dir, 'capi', '*cuda*.so'))
        if cuda_so_list:
            for so in cuda_so_list:
                try:
                    ctypes.CDLL(so)
                except OSError as err:
                    dl_error = str(err)
                    break
        else:
            dl_error = f"No CUDA provider shared object found in {ort_dir}/capi"
    except Exception as ex:
        dl_error = str(ex)

    return False, f"Providers: {providers}", dl_error
