"""Tests for SAM segmentation.

SAM requires the optional segment-geospatial dependency. These tests
verify the module structure and error handling without requiring the
full SAM model.
"""

import numpy as np
import pytest

from geobia.segmentation.sam import _check_sam_available, SAMSegmenter


sam_available = _check_sam_available()


class TestSAMAvailability:
    def test_check_function_returns_bool(self):
        result = _check_sam_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(sam_available, reason="samgeo is installed")
    def test_import_error_without_samgeo(self):
        with pytest.raises(ImportError, match="segment-geospatial"):
            SAMSegmenter()

    def test_param_schema(self):
        schema = SAMSegmenter.get_param_schema()
        assert "properties" in schema
        assert "model_type" in schema["properties"]
        assert "points_per_side" in schema["properties"]

    @pytest.mark.skipif(not sam_available, reason="samgeo not installed")
    def test_constructor_defaults(self):
        seg = SAMSegmenter()
        assert seg.model_type == "vit_h"
        assert seg.points_per_side == 32
        params = seg.get_params()
        assert params["algorithm"] == "sam"

    def test_registration_consistent_with_availability(self):
        from geobia.segmentation import list_methods, _REGISTRY

        methods = list_methods()
        # SAM is only registered if import succeeded at module load time
        assert ("sam" in methods) == ("sam" in _REGISTRY)
