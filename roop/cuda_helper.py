import os
import sys
import glob
import ctypes

def preload_cuda_libraries():
    """
    Preloads NVIDIA CUDA runtime, cuBLAS, and cuDNN shared libraries into the process global symbol table.
    On Linux / Google Colab environments, libraries like libcudart.so, libcublasLt.so, and libcudnn.so
    are frequently installed via pip into nvidia/*/lib or located in /usr/local/cuda*/lib64, but are not
    in the default ld.so search path. Preloading them with RTLD_GLOBAL allows onnxruntime's CUDA provider
    to link against them dynamically without failing.
    """
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
