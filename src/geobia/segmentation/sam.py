"""Segment Anything Model (SAM) segmentation via segment-geospatial.

Requires the optional 'sam' dependency:
    pip install geobia[sam]
"""

from __future__ import annotations

import numpy as np

from geobia.segmentation.base import BaseSegmenter


def _check_sam_available():
    try:
        import samgeo  # noqa: F401

        return True
    except ImportError:
        return False


class SAMSegmenter(BaseSegmenter):
    """Segment Anything Model segmentation via segment-geospatial.

    Uses SAM's automatic mask generation to segment imagery. Best suited
    for high-resolution imagery where objects have clear visual boundaries.

    Requires: pip install geobia[sam]
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
        device: str | None = None,
    ):
        if not _check_sam_available():
            raise ImportError(
                "segment-geospatial is required for SAM segmentation. "
                "Install with: pip install geobia[sam]"
            )
        self.model_type = model_type
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.device = device

    def segment(
        self,
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Segment using SAM automatic mask generation.

        Args:
            image: (bands, height, width) array. Uses first 3 bands as RGB.
            nodata_mask: Boolean mask, True where invalid.

        Returns:
            (height, width) int32 label array, 0=nodata, 1..N=segments.
        """
        from samgeo import SamGeo

        # SAM expects RGB uint8 image (height, width, 3)
        if image.shape[0] >= 3:
            rgb = np.moveaxis(image[:3], 0, -1)
        elif image.shape[0] == 1:
            rgb = np.stack([image[0]] * 3, axis=-1)
        else:
            # 2 bands: pad with zeros
            padded = np.zeros((3, image.shape[1], image.shape[2]), dtype=image.dtype)
            padded[: image.shape[0]] = image
            rgb = np.moveaxis(padded, 0, -1)

        # Scale to uint8 if needed
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            else:
                rgb = rgb.clip(0, 255).astype(np.uint8)

        sam_kwargs = {}
        if self.device is not None:
            sam_kwargs["device"] = self.device

        sam = SamGeo(
            model_type=self.model_type,
            automatic=True,
            **sam_kwargs,
        )

        sam.generate(
            rgb,
            output=None,
            foreground=True,
            unique=True,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

        # Extract masks and build label array
        masks = sam.masks
        h, w = image.shape[1], image.shape[2]
        labels = np.zeros((h, w), dtype=np.int32)

        if masks is not None:
            # Sort by area (largest first) so smaller segments overwrite
            mask_list = []
            for i in range(len(masks)):
                mask = masks[i]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask.squeeze()
                area = mask.sum()
                if area >= self.min_mask_region_area:
                    mask_list.append((area, mask))

            mask_list.sort(key=lambda x: -x[0])  # largest first

            for seg_id, (_, mask) in enumerate(mask_list, start=1):
                labels[mask > 0] = seg_id

        if nodata_mask is not None:
            labels[nodata_mask] = 0

        return labels

    def get_params(self) -> dict:
        return {
            "algorithm": "sam",
            "model_type": self.model_type,
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "min_mask_region_area": self.min_mask_region_area,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "default": "vit_h",
                    "enum": ["vit_h", "vit_l", "vit_b"],
                    "description": "SAM model variant",
                },
                "points_per_side": {
                    "type": "integer",
                    "default": 32,
                    "minimum": 1,
                    "description": "Density of automatic point grid",
                },
                "pred_iou_thresh": {
                    "type": "number",
                    "default": 0.88,
                    "description": "Confidence threshold for masks",
                },
                "stability_score_thresh": {
                    "type": "number",
                    "default": 0.95,
                    "description": "Stability threshold for masks",
                },
                "min_mask_region_area": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 0,
                    "description": "Minimum segment area in pixels",
                },
            },
        }
