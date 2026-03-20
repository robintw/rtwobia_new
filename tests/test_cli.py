"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner

from geobia.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def features_file(tmp_path, raster_file, labels_file, runner):
    """Create a features parquet by running extract."""
    output = str(tmp_path / "features.parquet")
    runner.invoke(cli, ["extract", raster_file, labels_file, "-o", output])
    return output


class TestSegmentCommand:
    def test_segment_slic(self, runner, raster_file, tmp_path):
        output = str(tmp_path / "segments.tif")
        result = runner.invoke(
            cli,
            [
                "segment",
                raster_file,
                "-o",
                output,
                "--method",
                "slic",
                "--n-segments",
                "20",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "segments written to" in result.output

    def test_segment_felzenszwalb(self, runner, raster_file, tmp_path):
        output = str(tmp_path / "segments.tif")
        result = runner.invoke(
            cli,
            [
                "segment",
                raster_file,
                "-o",
                output,
                "--method",
                "felzenszwalb",
                "--scale",
                "50",
            ],
        )
        assert result.exit_code == 0, result.output


class TestExtractCommand:
    def test_extract_features(self, runner, raster_file, labels_file, tmp_path):
        output = str(tmp_path / "features.parquet")
        result = runner.invoke(
            cli,
            [
                "extract",
                raster_file,
                labels_file,
                "-o",
                output,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "features" in result.output


class TestClassifyCommand:
    def test_classify_kmeans(self, runner, features_file, tmp_path):
        output = str(tmp_path / "classified.parquet")
        result = runner.invoke(
            cli,
            [
                "classify",
                features_file,
                "-o",
                output,
                "--method",
                "kmeans",
                "--n-clusters",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "classified" in result.output

    def test_classify_to_csv(self, runner, features_file, tmp_path):
        output = str(tmp_path / "classified.csv")
        result = runner.invoke(
            cli,
            [
                "classify",
                features_file,
                "-o",
                output,
                "--method",
                "kmeans",
                "--n-clusters",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_classify_supervised_requires_training(self, runner, features_file, tmp_path):
        output = str(tmp_path / "classified.parquet")
        result = runner.invoke(
            cli,
            [
                "classify",
                features_file,
                "-o",
                output,
                "--method",
                "random_forest",
            ],
        )
        assert result.exit_code != 0


class TestExportCommand:
    def test_export_gpkg(self, runner, labels_file, tmp_path):
        output = str(tmp_path / "export.gpkg")
        result = runner.invoke(
            cli,
            [
                "export",
                labels_file,
                "-o",
                output,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Exported" in result.output

    def test_export_with_features(self, runner, labels_file, features_file, tmp_path):
        output = str(tmp_path / "export.gpkg")
        result = runner.invoke(
            cli,
            [
                "export",
                labels_file,
                "-o",
                output,
                "--features",
                features_file,
            ],
        )
        assert result.exit_code == 0, result.output


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

    def test_info_parquet(self, runner, features_file):
        result = runner.invoke(cli, ["info", features_file])
        assert result.exit_code == 0
        assert "Segments:" in result.output
        assert "Features:" in result.output

    def test_info_unknown_type(self, runner, tmp_path):
        path = str(tmp_path / "unknown.xyz")
        with open(path, "w") as f:
            f.write("test")
        result = runner.invoke(cli, ["info", path])
        assert result.exit_code != 0


class TestCLIErrors:
    def test_segment_missing_output(self, runner, raster_file):
        result = runner.invoke(cli, ["segment", raster_file])
        assert result.exit_code != 0

    def test_segment_missing_input(self, runner, tmp_path):
        output = str(tmp_path / "segments.tif")
        result = runner.invoke(
            cli,
            [
                "segment",
                "/nonexistent/image.tif",
                "-o",
                output,
            ],
        )
        assert result.exit_code != 0

    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()
