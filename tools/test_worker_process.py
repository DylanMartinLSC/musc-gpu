"""Windows CPU-only tests for owned-worker tree cleanup; imports no torch."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from benchmark_process import run_worker, worker_ready

def alive(pid):
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    api.OpenProcess.restype=wintypes.HANDLE
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.OpenProcess(0x100000,False,pid)
    if not handle:
        return False
    try:
        return api.WaitForSingleObject(handle,0)==0x102
    finally:
        api.CloseHandle(handle)

def fixture(directory,token,delay):
    directory=Path(directory)
    worker_ready(directory,token)
    marker=directory/'descendant_pid.txt'
    subprocess.Popen([sys.executable,'-c',
        'import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(120)',
        str(marker)])
    deadline=time.monotonic()+3
    while not marker.exists():
        if time.monotonic()>deadline:
            raise RuntimeError('Fixture descendant did not start')
        time.sleep(.05)
    time.sleep(delay)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',nargs=3,metavar=('DIRECTORY','TOKEN','DELAY'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.fixture:
        fixture(args.fixture[0],args.fixture[1],float(args.fixture[2]))
        return
    if os.name!='nt':
        raise SystemExit('Windows-only CPU process-tree test')
    results=[]
    for name,delay,limit in (('normal_completion',.1,15),('timeout',120,4)):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            token=f'test-{name}-{os.getpid()}'
            command=[sys.executable,str(Path(__file__).resolve()),'--fixture',tmp,token,str(delay)]
            timed_out=False
            with (directory/'stdout.log').open('w') as log:
                try:
                    result=run_worker(command,Path(__file__).resolve().parent,log,limit,token,directory)
                    assert result.returncode==0
                except subprocess.TimeoutExpired:
                    timed_out=True
            assert timed_out==(name=='timeout')
            worker_pid=json.loads((directory/'worker_ready.json').read_text())['pid']
            child_pid=int((directory/'descendant_pid.txt').read_text())
            deadline=time.monotonic()+3
            while (alive(worker_pid) or alive(child_pid)) and time.monotonic()<deadline:
                time.sleep(.05)
            assert not alive(worker_pid) and not alive(child_pid), 'Owned worker or descendant survived job closure'
            results.append({'case':name,'pass':True,'worker_pid':worker_pid,'descendant_pid':child_pid,
                            'worker_alive_after':False,'descendant_alive_after':False,'cuda_work':False})
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
    print('PASS: normal completion and timeout terminate actual venv workers and descendants; no CUDA.')

if __name__=='__main__':
    main()
