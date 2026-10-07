# Alignment-correct virtual softmax lanes

Status: full bottle83 qualified and deployed by root. Verified3.836766x ORIGINAL;10x remains unmet.

Hypothesis: the accepted1370-column row softmax reserves1024 physical threads to preserve PyTorch's logical reduction tree. Four aligned logical lanes per256 physical threads can preserve every FP32 operation while improving GPU scheduling. This differs materially from historical rejected virtual lanes because it retains the now-established row-origin alignment offset and original32 virtual warp partial ordering. Register-reuse alone was rejected in the parent lane; this candidate starts again from deployed attention, including its original exponential expressions.

Actual path: model's24 attention modules identify retained block12; softmax intercept selects the existing qualified half/autocast/no-grad1370-square contiguous dispatch. Kernel launches one256-thread block per row. Logical lane tid+256*g loads first=logical_tid-(origin&7), second=first+1024. Four warp reductions preserve shuffle steps16/8/4/2/1 and place their results at physical_warp+8*g. Physical warp0 reduces the original32 ordered partials. Per-lane summation remains0+first_exp+second_exp; final division and exact half casts are unchanged. The23 discarded blocks retain half-only stores; block12 keeps FP32/FP16 stores. Caller stream and per-call aggregation factory are preserved.

CPU mapping checks cover16 row origins and all observed alignment residues. CPU fallback, global/instance restoration on success/exception and independent aggregation factory checks pass. NumPy exp is a placeholder in mapping analysis; GPU evidence establishes actual numerical agreement.

Real-weight full24 batch4 and batch1 checks show bitwise agreement for pooled output and all four requested patch outputs (layers6/12/18/24). Each call dispatches24 fused softmax calls,23 half-only. Batch1 and component candidate SHA256 agree exactly; source_receipt.json preserves candidate, adapter, harness and deployed attention hashes.

| Encoder screen | Incumbent seconds | Candidate seconds | Ratio |
|---|---:|---:|---:|
| Incumbent / candidate |3.496942|3.106379|1.125730x|
| Candidate / incumbent |3.523188|2.993705|1.176865x|

Each screen includes21 encoder batches with final batch3. First candidate pass includes118.946ms first-use NVRTC setup measured during correctness and charged to that pass; raw first candidate time2.987433s. Equal component peak allocation2955440128bytes. Model load is separate.

Bottle8 actual inference agrees bitwise in scores/maps and labels. First cold pair1.224988s/0.572311s gives2.140423x and is inflated by baseline first-use costs. The alternating repeat0.503715s/0.487047s gives1.034223x, exceeding the3% floor narrowly. Do not use the cold ratio as an expected gain. Candidate peak2782.833MiB. Full setup, loading, preprocessing, transfers, aggregation/scoring and output processing remain inside inference, with model load and metrics separate.

Environment RTX3090, PyTorch2.7.1+cu118, configured Python,518px CLIP, batch4, unchanged weights/layers/radii. Explicit root GPU grants, Local\\MuScGpuBenchmark mutex, supervisor checks and empty model API unload checks protect all runs. Component launch bounded480s; GPU slot released after batch1, whose post-check also records empty API.

Authoritative artifacts: probe_results.json, batch1_results.json, bottle8/results.json, cpu_results.json, mapping_results.json, source_receipt.json, process_receipt.json, logs. Commands:

```
& C:/Users/dylan/myenv/myenv/Scripts/python.exe experiments/parallel_20261004/round6/encoder_candidate/virtual_aligned/launch.py
& C:/Users/dylan/myenv/myenv/Scripts/python.exe experiments/parallel_20261004/round2/trial.py --candidate experiments/parallel_20261004/round6/encoder_candidate/virtual_aligned/adapter.py --images 8 --output experiments/parallel_20261004/round6/encoder_candidate/virtual_aligned/bottle8
& C:/Users/dylan/myenv/myenv/Scripts/python.exe experiments/parallel_20261004/round6/encoder_candidate/virtual_aligned/batch1.py --verified-offline-unload
```

build.py is a generator from the prior production; use the saved candidate to reproduce existing evidence. No scoring changes or approximation.

## Full83 acceptance and deployment

Two alternating complete-pool comparisons pass, including first-use setup with no discarded image warmup. Model loading and metrics are separate. Both maps and scores are bitwise identical to the prior incumbent. Original-relative maximum errors remain4.314954e-6 maps and2.086163e-7 scores; labels agree.

| Pass | Original seconds | Prior GPU seconds | Candidate seconds | Gain vs prior | Original/candidate |
|---|---:|---:|---:|---:|---:|
| Original / prior / candidate |37.619155|10.325768|9.757000|1.058293x|3.855607x|
| Candidate / prior / original |36.646450|10.432773|9.599306|1.086826x|3.817615x|

Medians:37.132802s original,10.379270s prior,9.678153s candidate. Verified3.836766x original and1.072443x paired incremental gain. Peak candidate allocation7309.460MiB. The paired10x threshold is3.713280s; another2.606360x gain is needed. Neither pass reaches10x.

Root ran the existing round2/trial.py with this adapter, --images83, --original and --output pointing to this lane's bottle83 directory. Authoritative evidence:bottle83/results.json. Source receipt matches all qualified candidate/harness hashes and previous deployed attention source. deploy.py preserved before_deployment/_attention_gpu.py, replaced only production attention, and verified CUDA source equality, function AST equality, CPU fallback/restoration and aggregation factory preservation without additional CUDA work. deployment_checks.json records hashes and the full-report digest.
