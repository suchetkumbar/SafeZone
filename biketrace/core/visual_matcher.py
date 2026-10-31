"""Visual feature extraction and similarity matching for identifying specific 2-wheelers."""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from typing import Optional, Tuple, List


def calculate_sharpness(image: np.ndarray) -> float:
    """
    Compute focus/sharpness score of an image using Laplacian variance.
    Higher score indicates sharper focus; lower indicates motion blur.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class VisualMatcher:
    """
    Extracts deep visual embeddings and HSV color signatures to identify a specific stolen bike
    from reference images or anchor crops.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Load lightweight MobileNetV3 backbone for feature extraction
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        full_model = models.mobilenet_v3_small(weights=weights)
        # Retain features + avgpool, drop final classifier
        self.encoder = full_model.features
        self.pool = full_model.avgpool
        self.encoder.eval()
        self.encoder.to(self.device)
        self.pool.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_deep_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 576-dim embedding vector."""
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros(576, dtype=np.float32)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.encoder(tensor)
            pooled = self.pool(feat).flatten().cpu().numpy()

        norm = np.linalg.norm(pooled)
        if norm > 1e-6:
            pooled /= norm
        return pooled

    def extract_color_signature(self, image_bgr: np.ndarray) -> np.ndarray:
        """Extract 3D HSV color histogram normalized to unit sum."""
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros(24 * 8 * 8, dtype=np.float32)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Hue: 24 bins (0..180), Sat: 8 bins, Val: 8 bins
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [24, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    def extract_composite_features(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both deep embedding and color signature."""
        return self.extract_deep_embedding(image_bgr), self.extract_color_signature(image_bgr)

    def compute_similarity(
        self,
        ref_deep: np.ndarray,
        ref_color: np.ndarray,
        cand_deep: np.ndarray,
        cand_color: np.ndarray,
        deep_weight: float = 0.65
    ) -> float:
        """
        Compute combined similarity score between reference and candidate image [0.0 .. 1.0].
        """
        # Cosine similarity for deep embedding
        sim_deep = float(np.dot(ref_deep, cand_deep))
        sim_deep = max(0.0, min(1.0, (sim_deep + 1.0) / 2.0))

        # Histogram intersection or correlation for color
        sim_color = float(cv2.compareHist(
            ref_color.reshape(-1, 1),
            cand_color.reshape(-1, 1),
            cv2.HISTCMP_CORREL
        ))
        sim_color = max(0.0, min(1.0, (sim_color + 1.0) / 2.0))

        return (deep_weight * sim_deep) + ((1.0 - deep_weight) * sim_color)
