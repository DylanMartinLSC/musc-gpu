# Attribution and third-party notices

This is an unofficial derivative of [MuSc](https://github.com/xrli-U/MuSc),
by **Xurui Li, Ziming Huang, Feng Xue, and Yu Zhou**. The original method and
paper results belong to those authors. The original MIT license and
Copyright (c) 2024 Xurui Li notice are preserved in `LICENSE`.

The local optimization work adds exact GPU scoring, direct pooling, ordered
threaded loading, per-division normalization reuse, and attention kernels.
It does not introduce new trained weights or claim authorship of MuSc.
Release packaging relocates the runtime import and adds platform selection,
documentation, evidence graphs, and a source manifest.

Bundled backbone sources retain their original headers:

| Source | Upstream attribution |
|---|---|
| `models/backbone/open_clip/` | [OpenCLIP](https://github.com/mlfoundations/open_clip); MIT. MuSc's modified implementation is retained. |
| CLIP-derived model/tokenizer code and BPE vocabulary | [OpenAI CLIP](https://github.com/openai/CLIP); MIT, Copyright (c) 2021 OpenAI. |
| `models/backbone/vision_transformer.py` | [DINO](https://github.com/facebookresearch/dino); Apache-2.0; Facebook, Inc. and affiliates. |
| `models/backbone/dino_vision_transformer.py`, `models/backbone/dinov2/` | [DINOv2](https://github.com/facebookresearch/dinov2); retain Meta Platforms, Inc. and affiliates notices and applicable upstream license. |

MuSc upstream also acknowledges [PatchCore](https://github.com/amazon-science/patchcore-inspection)
and [APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN). That attribution is retained here.
Model weights and datasets are downloaded separately and retain their own
terms; complete datasets and model weights are not distributed here. Third-party licenses are
included under `licenses/`; they do not become MIT merely because MuSc is MIT.

## MVTec AD example figure

`benchmarks/segmentation_usefulness/examples.png` adapts three images and their
defect annotations from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
MVTec Software GmbH. The dataset and this adapted figure are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), separately
from the code's MIT license. Changes: resizing, colored annotation/prediction
overlays, labels, and montage layout. Original image identifiers are recorded
in `benchmarks/segmentation_usefulness/examples.json`.
