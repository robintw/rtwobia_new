"""Tests for the I/O layer."""

import numpy as np

from geobia.io.raster import read_raster, read_raster_windows, write_raster


class TestReadRaster:
    def test_reads_image_shape(self, raster_file):
        image, meta = read_raster(raster_file)
        assert image.shape == (4, 100, 100)

    def test_reads_metadata(self, raster_file):
        _, meta = read_raster(raster_file)
        assert meta["width"] == 100
        assert meta["height"] == 100
        assert meta["count"] == 4
        assert meta["crs"] is not None

    def test_reads_correct_values(self, raster_file, synthetic_image):
        image, _ = read_raster(raster_file)
        np.testing.assert_array_almost_equal(image, synthetic_image, decimal=5)


class TestWriteRaster:
    def test_roundtrip(self, tmp_path, synthetic_image, synthetic_meta):
        path = str(tmp_path / "out.tif")
        write_raster(path, synthetic_image, synthetic_meta)
        image, meta = read_raster(path)
        np.testing.assert_array_almost_equal(image, synthetic_image, decimal=5)
        assert meta["width"] == 100

    def test_writes_2d_array(self, tmp_path, synthetic_meta):
        data = np.ones((100, 100), dtype=np.int32) * 5
        path = str(tmp_path / "labels.tif")
        write_raster(path, data, synthetic_meta, dtype="int32")
        result, _ = read_raster(path)
        assert result.shape == (1, 100, 100)
        assert result[0, 0, 0] == 5


class TestReadRasterWindows:
    def test_tiles_cover_full_image(self, raster_file):
        tiles = list(read_raster_windows(raster_file, tile_size=50, overlap=0))
        # 100x100 image with 50px tiles = 4 tiles
        assert len(tiles) == 4

    def test_tile_has_correct_shape(self, raster_file):
        tiles = list(read_raster_windows(raster_file, tile_size=60, overlap=0))
        tile, meta, window = tiles[0]
        assert tile.shape[0] == 4  # bands
        assert tile.shape[1] == 60
        assert tile.shape[2] == 60

    def test_overlap_produces_more_tiles(self, raster_file):
        no_overlap = list(read_raster_windows(raster_file, tile_size=50, overlap=0))
        with_overlap = list(read_raster_windows(raster_file, tile_size=50, overlap=10))
        assert len(with_overlap) >= len(no_overlap)
