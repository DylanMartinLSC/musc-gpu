"""Small Windows NVRTC/driver bridge using the CUDA DLLs shipped with PyTorch."""
import ctypes as ct
import os
from pathlib import Path
import torch

_dll_handles = []
_libraries = None
_modules = {}


def _libraries_for_cuda():
    global _libraries
    if _libraries is not None:
        return _libraries
    if os.name != 'nt':
        raise RuntimeError('This research bridge currently targets native Windows.')
    lib = Path(torch.__file__).parent / 'lib'
    _dll_handles.append(os.add_dll_directory(str(lib)))
    paths = sorted(lib.glob('nvrtc64*.dll'))
    if not paths:
        raise RuntimeError('PyTorch NVRTC DLL not found.')
    nvrtc = ct.WinDLL(str(paths[-1]))
    driver = ct.WinDLL('nvcuda.dll')
    prototypes = {
        'nvrtcCreateProgram': [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p],
        'nvrtcCompileProgram': [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)],
        'nvrtcGetProgramLogSize': [ct.c_void_p, ct.POINTER(ct.c_size_t)],
        'nvrtcGetProgramLog': [ct.c_void_p, ct.c_void_p],
        'nvrtcGetPTXSize': [ct.c_void_p, ct.POINTER(ct.c_size_t)],
        'nvrtcGetPTX': [ct.c_void_p, ct.c_void_p],
        'nvrtcDestroyProgram': [ct.POINTER(ct.c_void_p)],
    }
    for name, args in prototypes.items():
        function = getattr(nvrtc, name)
        function.argtypes, function.restype = args, ct.c_int
    driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
    driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
    driver.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
    driver.cuGetErrorName.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
    for name in ['cuModuleLoadData', 'cuModuleGetFunction', 'cuLaunchKernel', 'cuGetErrorName']:
        getattr(driver, name).restype = ct.c_int
    _libraries = nvrtc, driver
    return _libraries


def _check(result, operation, driver=None):
    if result:
        detail = str(result)
        if driver is not None:
            name = ct.c_char_p()
            driver.cuGetErrorName(result, ct.byref(name))
            if name.value:
                detail = name.value.decode()
        raise RuntimeError(f'{operation}: {detail}')


def compile_kernels(source, names, device):
    key = (source, tuple(names), int(device))
    if key in _modules:
        return _modules[key][1]
    nvrtc, driver = _libraries_for_cuda()
    with torch.cuda.device(device):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability(device)
        program = ct.c_void_p()
        _check(nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b'musc.cu', 0, None, None), 'nvrtcCreateProgram')
        try:
            options = (ct.c_char_p * 2)(f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--std=c++14')
            status = nvrtc.nvrtcCompileProgram(program, 2, options)
            if status:
                size = ct.c_size_t()
                nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
                log = ct.create_string_buffer(size.value or 1)
                nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(f'NVRTC compilation failed: {log.value.decode(errors="replace")}')
            size = ct.c_size_t()
            _check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), 'nvrtcGetPTXSize')
            ptx = ct.create_string_buffer(size.value)
            _check(nvrtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
        finally:
            nvrtc.nvrtcDestroyProgram(ct.byref(program))
        module = ct.c_void_p()
        _check(driver.cuModuleLoadData(ct.byref(module), ptx), 'cuModuleLoadData', driver)
        functions = {}
        for name in names:
            function = ct.c_void_p()
            _check(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()), 'cuModuleGetFunction', driver)
            functions[name] = function
        # Retain modules while asynchronous work may reference them.
        _modules[key] = module, functions
        return functions


def launch(function, grid, block, tensors, integers, device):
    _, driver = _libraries_for_cuda()
    arguments = [ct.c_void_p(t.data_ptr()) for t in tensors] + [ct.c_int(v) for v in integers]
    pointers = (ct.c_void_p * len(arguments))(*[ct.cast(ct.byref(a), ct.c_void_p) for a in arguments])
    stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    _check(driver.cuLaunchKernel(function, *grid, *block, 0, stream, pointers, None), 'cuLaunchKernel', driver)
