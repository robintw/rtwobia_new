"""Command-line interface for geobia."""

from __future__ import annotations

import json
import sys

import click


@click.group()
@click.version_option(package_name="geobia")
def cli():
    """geobia: Object-Based Image Analysis for geospatial imagery."""


@cli.command()
@click.argument("input_image")
@click.option("-o", "--output", required=True, help="Output segment labels file (GeoTIFF)")
@click.option("--method", default="slic", type=click.Choice(["slic", "felzenszwalb", "shepherd"]),
              show_default=True, help="Segmentation algorithm")
@click.option("--n-segments", type=int, default=500, show_default=True, help="Number of segments (SLIC)")
@click.option("--compactness", type=float, default=10.0, show_default=True, help="Compactness (SLIC)")
@click.option("--scale", type=float, default=100.0, show_default=True, help="Scale parameter (Felzenszwalb)")
@click.option("--min-size", type=int, default=50, show_default=True, help="Minimum segment size (Felzenszwalb/Shepherd)")
@click.option("--num-clusters", type=int, default=60, show_default=True, help="K-means clusters (Shepherd)")
@click.option("--dist-thres", type=float, default=100.0, show_default=True, help="Spectral distance threshold (Shepherd)")
@click.option("--sampling", type=int, default=100, show_default=True, help="Subsampling rate (Shepherd)")
@click.option("--sigma", type=float, default=None, help="Gaussian smoothing sigma")
@click.option("--tiled", is_flag=True, help="Use tiled processing for large images")
@click.option("--tile-size", type=int, default=2048, show_default=True, help="Tile size for tiled processing")
def segment(input_image, output, method, n_segments, compactness, scale, min_size,
            num_clusters, dist_thres, sampling, sigma, tiled, tile_size):
    """Segment an image into objects."""
    from geobia.io.raster import read_raster, write_raster
    from geobia.segmentation import segment as do_segment, segment_tiled

    params = {}
    if method == "slic":
        params["n_segments"] = n_segments
        params["compactness"] = compactness
        if sigma is not None:
            params["sigma"] = sigma
    elif method == "felzenszwalb":
        params["scale"] = scale
        params["min_size"] = min_size
        if sigma is not None:
            params["sigma"] = sigma
    elif method == "shepherd":
        params["num_clusters"] = num_clusters
        params["min_n_pxls"] = min_size
        params["dist_thres"] = dist_thres
        params["sampling"] = sampling

    if tiled:
        click.echo(f"Segmenting {input_image} with {method} (tiled, tile_size={tile_size})...")
        labels = segment_tiled(input_image, method=method, tile_size=tile_size,
                               overlap=128, output_path=output, **params)
    else:
        click.echo(f"Segmenting {input_image} with {method}...")
        image, meta = read_raster(input_image)
        labels = do_segment(image, method=method, **params)
        write_raster(output, labels, meta, dtype="int32")

    n = len(set(labels.flat)) - (1 if 0 in labels else 0)
    click.echo(f"Done. {n} segments written to {output}")


@cli.command()
@click.argument("input_image")
@click.argument("segments")
@click.option("-o", "--output", required=True, help="Output feature file (Parquet)")
@click.option("--spectral/--no-spectral", default=True, show_default=True, help="Extract spectral features")
@click.option("--geometry/--no-geometry", default=True, show_default=True, help="Extract geometric features")
@click.option("--texture/--no-texture", default=False, show_default=True, help="Extract GLCM texture features")
@click.option("--band-names", type=str, default=None,
              help="Comma-separated band names (e.g., red,green,blue,nir)")
def extract(input_image, segments, output, spectral, geometry, texture, band_names):
    """Extract features from segmented image."""
    from geobia.io.raster import read_raster
    from geobia.features import extract as do_extract

    click.echo(f"Extracting features from {input_image}...")

    image, meta = read_raster(input_image)
    seg_data, _ = read_raster(segments)
    labels = seg_data[0]

    categories = []
    if spectral:
        categories.append("spectral")
    if geometry:
        categories.append("geometry")
    if texture:
        categories.append("texture")

    kwargs = {}
    if band_names:
        names = band_names.split(",")
        kwargs["band_names"] = {name.strip(): i for i, name in enumerate(names)}

    pixel_size = abs(meta["transform"].a) if meta.get("transform") else None
    if pixel_size:
        kwargs["pixel_size"] = pixel_size

    features = do_extract(image, labels, categories=categories, **kwargs)
    features.to_parquet(output)

    click.echo(f"Done. {len(features)} segments, {len(features.columns)} features -> {output}")


@cli.command()
@click.argument("features_file")
@click.option("-o", "--output", required=True, help="Output classified file (GeoPackage or Parquet)")
@click.option("--method", default="random_forest",
              type=click.Choice(["random_forest", "kmeans", "gmm", "dbscan"]),
              show_default=True, help="Classification method")
@click.option("--training", type=click.Path(exists=True), default=None,
              help="Training samples file (required for supervised)")
@click.option("--segments", type=click.Path(exists=True), default=None,
              help="Segment labels raster (for mapping training samples)")
@click.option("--n-clusters", type=int, default=8, show_default=True, help="Number of clusters (K-Means)")
@click.option("--n-estimators", type=int, default=100, show_default=True, help="Number of trees (Random Forest)")
def classify(features_file, output, method, training, segments, n_clusters, n_estimators):
    """Classify segments using their features."""
    import pandas as pd
    from geobia.classification import classify as do_classify

    click.echo(f"Classifying with {method}...")

    features = pd.read_parquet(features_file)

    params = {}
    training_labels = None

    if method == "kmeans":
        params["n_clusters"] = n_clusters
    elif method == "random_forest":
        params["n_estimators"] = n_estimators
        if training is None:
            click.echo("Error: --training required for supervised classification", err=True)
            sys.exit(1)

        if segments is None:
            click.echo("Error: --segments required to map training samples", err=True)
            sys.exit(1)

        from geobia.io.raster import read_raster
        from geobia.io.vector import read_training_samples

        seg_data, meta = read_raster(segments)
        labels = seg_data[0]
        training_labels = read_training_samples(training, labels, meta)

    predictions = do_classify(features, method=method, training_labels=training_labels, **params)

    result = features.copy()
    result["class_label"] = predictions

    if output.endswith(".parquet"):
        result.to_parquet(output)
    else:
        result.to_csv(output.replace(".gpkg", ".csv"))

    click.echo(f"Done. {len(predictions)} segments classified -> {output}")


@cli.command()
@click.argument("input_file")
def info(input_file):
    """Show information about a geobia dataset."""
    import numpy as np

    if input_file.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(input_file) as ds:
            click.echo(f"File: {input_file}")
            click.echo(f"Size: {ds.width} x {ds.height}")
            click.echo(f"Bands: {ds.count}")
            click.echo(f"Dtype: {ds.dtypes[0]}")
            click.echo(f"CRS: {ds.crs}")
            click.echo(f"Resolution: {ds.res}")
            click.echo(f"Bounds: {ds.bounds}")

            if ds.dtypes[0] in ("int32", "uint32", "int16", "uint16"):
                data = ds.read(1)
                unique = np.unique(data)
                n_segments = int((unique > 0).sum())
                if n_segments > 0:
                    click.echo(f"Segments: {n_segments}")

    elif input_file.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(input_file)
        click.echo(f"File: {input_file}")
        click.echo(f"Segments: {len(df)}")
        click.echo(f"Features: {len(df.columns)}")
        click.echo(f"Feature names: {', '.join(df.columns[:20])}")
        if len(df.columns) > 20:
            click.echo(f"  ... and {len(df.columns) - 20} more")

    elif input_file.endswith(".gpkg"):
        import geopandas as gpd
        gdf = gpd.read_file(input_file)
        click.echo(f"File: {input_file}")
        click.echo(f"Features: {len(gdf)}")
        click.echo(f"Columns: {', '.join(gdf.columns)}")
        click.echo(f"CRS: {gdf.crs}")

    else:
        click.echo(f"Unknown file type: {input_file}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("segments")
@click.option("-o", "--output", required=True, help="Output vector file (GeoPackage)")
@click.option("--features", type=click.Path(exists=True), default=None,
              help="Feature Parquet file to include as attributes")
@click.option("--classification", type=click.Path(exists=True), default=None,
              help="Classification Parquet file to include")
def export(segments, output, features, classification):
    """Export segments as vector with optional attributes."""
    import pandas as pd
    from geobia.io.raster import read_raster
    from geobia.io.vector import write_vector

    click.echo(f"Exporting {segments} to {output}...")

    seg_data, meta = read_raster(segments)
    labels = seg_data[0]

    attributes = None
    if features:
        attributes = pd.read_parquet(features)
    if classification:
        clf_data = pd.read_parquet(classification)
        if attributes is not None:
            attributes = attributes.join(clf_data[["class_label"]], how="left")
        else:
            attributes = clf_data

    write_vector(output, labels, attributes=attributes, meta=meta)
    click.echo(f"Done. Exported to {output}")
