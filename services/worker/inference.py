from __future__ import annotations

import logging
import threading
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import models.backbone._backbones as _backbones
import models.backbone.open_clip as open_clip
from models.modules._LNAMD import LNAMD
from models.modules._MSM import compute_scores_fast

from .schemas import StreamConfig

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _default_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class StreamingMuSc:
    def __init__(self, config: StreamConfig, device_override: str | None = None) -> None:
        self.config = config
        device_str = device_override or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.model_name = config.backbone_name
        self.image_size = config.image_size
        self.features_list = [layer + 1 for layer in config.feature_layers]
        self.r_list = config.r_list
        self.pretrained = config.pretrained
        self.vis_type = config.vis_type
        self._lock = threading.Lock()

        self.clip_model = None
        self.dino_model = None
        self.preprocess = None
        self._load_backbone()

    def _load_backbone(self) -> None:
        if "dino" in self.model_name:
            self.dino_model = _backbones.load(self.model_name)
            self.dino_model.to(self.device)
            self.dino_model.eval()
            self.preprocess = _default_transform(self.image_size)
        else:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, self.image_size, pretrained=self.pretrained
            )
            clip_model.to(self.device)
            clip_model.eval()
            self.clip_model = clip_model
            self.preprocess = preprocess

        if self.preprocess is None:
            self.preprocess = _default_transform(self.image_size)

    def _prepare_batch(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = self.preprocess(image)
            tensors.append(tensor)
        return torch.stack(tensors)

    def _mutual_scoring(self, features: torch.Tensor) -> torch.Tensor:
        scores = []
        for i in range(features.shape[0]):
            scores.append(
                compute_scores_fast(features, i, self.device, topmin_min=0, topmin_max=0.3)
            )
        return torch.stack(scores, dim=0)

    def process_batch(self, frames: Sequence[np.ndarray]) -> Tuple[List[float], List[bytes]]:
        if not frames:
            return [], []

        with self._lock:
            batch = self._prepare_batch(frames)
            batch = batch.to(self.device)
            patch_tokens_list: List[List[torch.Tensor]] = []
            autocast_enabled = self.device.type == "cuda"
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
                if self.clip_model is not None:
                    image_features, patch_tokens = self.clip_model.encode_image(batch, self.features_list)
                    patch_tokens = [patch_tokens[idx].cpu() for idx in range(len(self.features_list))]
                else:
                    if "dinov2" in self.model_name:
                        patch_tokens = self.dino_model.get_intermediate_layers(
                            x=batch,
                            n=[layer - 1 for layer in self.features_list],
                            return_class_token=False,
                        )
                        image_features = self.dino_model(batch)
                        patch_tokens = [patch_tokens[idx].cpu() for idx in range(len(self.features_list))]
                        fake_cls = [torch.zeros_like(p)[:, 0:1, :] for p in patch_tokens]
                        patch_tokens = [torch.cat([fake_cls[i], patch_tokens[i]], dim=1) for i in range(len(patch_tokens))]
                    else:
                        patch_tokens_all = self.dino_model.get_intermediate_layers(
                            x=batch, n=max(self.features_list)
                        )
                        image_features = self.dino_model(batch)
                        patch_tokens = [patch_tokens_all[layer - 1].cpu() for layer in self.features_list]

            patch_tokens_list.append(patch_tokens)

            feature_dim = patch_tokens_list[0][0].shape[-1]
            anomaly_maps_r: List[torch.Tensor] = []

            for r in self.r_list:
                lnamd = LNAMD(
                    device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features_list
                )
                z_layers: dict[str, List[torch.Tensor]] = {}
                for patch_tokens in patch_tokens_list:
                    patch_tokens_device = [tensor.to(self.device) for tensor in patch_tokens]
                    features = lnamd._embed(patch_tokens_device).to(self.device)
                    features = features / (features.norm(dim=-1, keepdim=True) + 1e-10)
                    for layer_idx in range(len(self.features_list)):
                        key = str(layer_idx)
                        if key not in z_layers:
                            z_layers[key] = []
                        z_layers[key].append(features[:, :, layer_idx, :])
                anomaly_maps_l: List[torch.Tensor] = []
                for layer_key, layer_features in z_layers.items():
                    z = torch.cat(layer_features, dim=0)
                    scores = self._mutual_scoring(z)
                    anomaly_maps_l.append(scores.cpu())
                layer_mean = torch.mean(torch.stack(anomaly_maps_l, dim=0), dim=0)
                anomaly_maps_r.append(layer_mean)

            anomaly_maps = torch.mean(torch.stack(anomaly_maps_r, dim=0), dim=0)
            patch_count = anomaly_maps.shape[1]
            side = int(np.sqrt(patch_count))
            anomaly_maps = anomaly_maps.view(anomaly_maps.shape[0], 1, side, side)
            anomaly_maps = F.interpolate(anomaly_maps, size=self.image_size, mode="bilinear", align_corners=True)
            anomaly_maps_np = anomaly_maps.squeeze(1).cpu().numpy()

        scores: List[float] = []
        heatmap_bytes: List[bytes] = []
        for frame, anomaly_map in zip(frames, anomaly_maps_np):
            normalized = self._normalize_map(anomaly_map)
            scores.append(float(normalized.max()))
            resized = cv2.resize((normalized * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]))
            heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            ok, buffer = cv2.imencode(".png", heatmap)
            if not ok:
                raise RuntimeError("Failed to encode heatmap image")
            heatmap_bytes.append(buffer.tobytes())
        return scores, heatmap_bytes

    def _normalize_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        minimum = anomaly_map.min()
        maximum = anomaly_map.max()
        denominator = maximum - minimum
        if denominator < 1e-8:
            return np.zeros_like(anomaly_map)
        normalized = (anomaly_map - minimum) / denominator
        return np.clip(normalized, 0.0, 1.0)
