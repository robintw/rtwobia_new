"""Tests for CLI commands."""

import numpy as np
import pytest
from click.testing import CliRunner

from geobia.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestSegmentCommand:
    def test_segment_slic(self, runner, raster_file, tmp_path):
        output = str(tmp_path / "segments.tif")
        result = runner.invoke(cli, [
            "segment", raster_file,
            "-o", output,
            "--method", "slic",
            "--n-segments", "20",
        ])
        assert result.exit_code == 0, result.output
        assert "segments written to" in result.output

    def test_segment_felzenszwalb(self, runner, raster_file, tmp_path):
        output = str(tmp_path / "segments.tif")
        result = runner.invoke(cli, [
            "segment", raster_file,
            "-o", output,
            "--method", "felzenszwalb",
            "--scale", "50",
        ])
        assert result.exit_code == 0, result.output


class TestExtractCommand:
    def test_extract_features(self, runner, raster_file, labels_file, tmp_path):
        output = str(tmp_path / "features.parquet")
        result = runner.invoke(cli, [
            "extract", raster_file, labels_file,
            "-o", output,
        ])
        assert result.exit_code == 0, result.output
        assert "features" in result.output


class TestInfoCommand:
    def test_info_raster(self, runner, raster_file):
        result = runner.invoke(cli, ["info", raster_file])
        assert result.exit_code == 0
        assert "Bands: 4" in result.output
        assert "100 x 100" in result.output

    def test_info_labels(self, runner, labels_file):
        result = runner.invoke(cli, ["info", labels_file])
        assert result.exit_code == 0
        assert "Segments: 4" in result.output
