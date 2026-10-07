"""Own a Windows worker's process tree before permitting CUDA initialization."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time

class BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
        ('flags', wintypes.DWORD), ('min_working_set', ctypes.c_size_t),
        ('max_working_set', ctypes.c_size_t), ('active_processes', wintypes.DWORD),
        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]

class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
        ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]

class ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IoCounters),
        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
        ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

def worker_ready(directory, token, timeout=60):
    """Call before importing torch. Wait until the parent owns this actual PID."""
    directory = Path(directory)
    ready = directory/'worker_ready.json'
    temporary = directory/'worker_ready.tmp'
    temporary.write_text(json.dumps({'pid': os.getpid(), 'token': token}), encoding='utf-8')
    temporary.replace(ready)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        go = directory/'worker_go.json'
        if go.exists():
            if json.loads(go.read_text())['token'] != token:
                raise RuntimeError('Worker start token mismatch')
            return
        time.sleep(.05)
    raise TimeoutError('Parent did not assign worker to its Windows Job Object')

def run_worker(command, cwd, log, timeout, job_token, trial_dir):
    """Run a bounded worker; Job Object closure kills its assigned descendants."""
    if os.name != 'nt':
        raise RuntimeError('Windows Job Object worker runner requires native Windows')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        'OpenProcess': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        'TerminateProcess': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name,(args,result) in signatures.items():
        function=getattr(api,name)
        function.argtypes,function.restype=args,result
    job=api.CreateJobObjectW(None,None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    process=None
    actual_handle=None
    assigned=False
    try:
        limits=ExtendedLimits()
        limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        start=time.monotonic()
        deadline=start+timeout
        process=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,MUSC_BENCHMARK_WORKER_TOKEN=job_token))
        ready=Path(trial_dir)/'worker_ready.json'
        while not ready.exists():
            if process.poll() is not None:
                return subprocess.CompletedProcess(command,process.returncode)
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command,timeout)
            time.sleep(.05)
        receipt=json.loads(ready.read_text())
        if receipt['token'] != job_token:
            raise RuntimeError('Worker readiness token mismatch')
        # This PID is the actual interpreter, rather than the venv redirector.
        actual_handle=api.OpenProcess(0x0100 | 0x0001,False,int(receipt['pid']))
        if not actual_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if not api.AssignProcessToJobObject(job,actual_handle):
            raise ctypes.WinError(ctypes.get_last_error())
        assigned=True
        temporary=Path(trial_dir)/'worker_go.tmp'
        temporary.write_text(json.dumps({'token':job_token}),encoding='utf-8')
        temporary.replace(Path(trial_dir)/'worker_go.json')
        remaining=deadline-time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command,timeout)
        code=process.wait(timeout=remaining)
        return subprocess.CompletedProcess(command,code)
    finally:
        # Close the last job handle even on cancellation or timeout. Every
        # process assigned before the go signal is terminated by Windows.
        api.CloseHandle(job)
        if actual_handle:
            if not assigned:
                api.TerminateProcess(actual_handle,1)  # our verified waiting worker only
            api.CloseHandle(actual_handle)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10 if assigned else .2)
            except subprocess.TimeoutExpired:
                # Before readiness there can be a venv redirector child, but
                # worker_ready prevents any CUDA work until job assignment.
                # Target only the exact process tree this runner created.
                killed=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                    capture_output=True,text=True,timeout=10)
                if killed.returncode and process.poll() is None:
                    raise RuntimeError(f'Failed to terminate owned launcher tree: {killed.stderr}')
                process.wait(timeout=10)
