"""
Fast preprocessing: Convert .tiff files directly to .npy cache format.

This skips CSV files entirely and writes directly to the numpy format
that the dataset uses, making preprocessing much faster.

For each municipality:
- Load all .tiff files (all tiles, all months)
- Extract each pixel's time series (6 months)
- Combine all tiles and save as single .npy file per municipality
- No index file needed - municipality code is in the filename
"""

import time
from collections import defaultdict
from multiprocessing import Manager, Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# Configuration
TIFF_ROOT_DIR = Path("files/gee_images")  # Root directory with municipality subfolders
OUTPUT_DIR = Path("files/yield_dataset")  # Where to save data
YIELD_CSV = Path("files/yield_data.csv")  # Municipality yield targets (to be created)

# Band names (10 spectral bands)
BANDNAMES = [
    "blue",
    "green",
    "red",
    "red1",
    "red2",
    "red3",
    "nir",
    "red4",
    "swir1",
    "swir2",
]


def parse_filename(filename):
    """
    Parse filename: 4100103_2023-01_tile_00_01.tif

    Returns:
        municipality_code, year, month, tile_x, tile_y
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    municipality_code = parts[0]
    year_month = parts[1]  # 2023-01
    year, month = year_month.split("-")
    tile_x = parts[3]  # 00
    tile_y = parts[4]  # 01

    return municipality_code, int(year), int(month), tile_x, tile_y


def process_tile(args):
    """
    Process a single tile (for parallel processing).
    Extracts all pixels and saves them to a single .npy file.

    Args:
        args: Tuple of (municipality_code, tile_x, tile_y, tile_files_by_month,
                       output_dir, progress_counter, progress_lock)

    Returns:
        tile_data: Array of pixel data [num_pixels, num_months, 11]
        num_pixels: Number of pixels processed
        npy_path: Path to the saved temporary .npy file (for combining later)
    """
    (
        municipality_code,
        tile_x,
        tile_y,
        tile_files_by_month,
        output_dir,
        progress_counter,
        progress_lock,
        expected_months,
    ) = args

    muni_dir = output_dir / municipality_code
    muni_dir.mkdir(parents=True, exist_ok=True)

    # Get height/width from first available file
    if len(tile_files_by_month) > 0:
        first_file = list(tile_files_by_month.values())[0]
        with rasterio.open(first_file) as src:
            height, width = src.height, src.width
    else:
        # No files for this tile, return empty array with correct shape
        empty_data = np.array([], dtype=np.float32).reshape(0, len(expected_months), 11)
        npy_filename = muni_dir / f"{municipality_code}_tile_{tile_x}_{tile_y}.npy"
        np.save(npy_filename, empty_data)
        return empty_data, 0, str(npy_filename)

    # Map month to approximate day of year
    month_to_doy = {1: 1, 2: 32, 3: 60, 4: 91, 5: 121, 6: 152}

    # OPTIMIZATION: Load all TIFF files into memory once
    # This avoids opening/closing files for each pixel
    tiff_data = {}  # {month: [10, height, width] array}
    nodata_values = {}  # {month: nodata value}

    # Load data for all expected months (use zeros if month missing for this tile)
    for month in expected_months:
        if month in tile_files_by_month:
            tiff_file = tile_files_by_month[month]
            try:
                with rasterio.open(tiff_file) as src:
                    # Read all bands at once: [num_bands, height, width]
                    # Cast to float32 immediately to save memory (50% reduction vs float64)
                    pixel_data = src.read()[:10].astype(
                        np.float32
                    )  # Only first 10 bands
                    tiff_data[month] = pixel_data
                    nodata_values[month] = src.nodata
            except Exception as e:
                # If file fails to load, create zeros array
                tiff_data[month] = np.zeros((10, height, width), dtype=np.float32)
                nodata_values[month] = None
        else:
            # Month missing for this tile, use zeros
            tiff_data[month] = np.zeros((10, height, width), dtype=np.float32)
            nodata_values[month] = None

    # Pre-allocate arrays for all pixels
    # Each pixel: [num_months, 11] (10 bands + doy)
    max_months = len(tile_files_by_month)
    all_pixel_data = []  # List of arrays, one per pixel
    pixel_metadata = []

    # Extract all pixels from this tile (now using pre-loaded data)
    total_pixels_in_tile = height * width
    pixels_processed_in_tile = 0

    for row in range(height):
        for col in range(width):
            timeseries = []
            doys = []

            for month in expected_months:
                pixel_data = tiff_data[month]  # [10, height, width]
                pixel_values = pixel_data[:, row, col]  # [10]

                # Check for nodata
                if nodata_values[month] is not None and np.all(
                    pixel_values == nodata_values[month]
                ):
                    pixel_values = np.zeros(10, dtype=np.float32)

                timeseries.append(pixel_values)
                doys.append(month_to_doy.get(month, month * 30))

            # Explicitly use float32 for spectral bands (memory optimization)
            timeseries = np.array(timeseries, dtype=np.float32)
            # DOY stored as uint8 (values 0-255, 75% memory reduction vs float32)
            doys = np.array(doys, dtype=np.uint8)

            # Filter out pixels that are always zero
            if np.all(timeseries == 0):
                pixels_processed_in_tile += 1
                # Update progress counter even for skipped pixels (every 1000)
                if (
                    progress_counter is not None
                    and pixels_processed_in_tile % 1000 == 0
                ):
                    with progress_lock:
                        progress_counter.value += 1000
                continue

            # Filter out pixels that have data in 2 months or fewer
            # Count months with non-zero data (check if any band has non-zero value)
            months_with_data = np.sum(np.any(timeseries != 0, axis=1))
            if months_with_data <= 2:
                pixels_processed_in_tile += 1
                # Update progress counter even for skipped pixels (every 1000)
                if (
                    progress_counter is not None
                    and pixels_processed_in_tile % 1000 == 0
                ):
                    with progress_lock:
                        progress_counter.value += 1000
                continue

            # Store spectral bands and DOY separately to maintain dtype efficiency
            # We'll combine them when saving using structured array
            all_pixel_data.append((timeseries, doys))

            pixels_processed_in_tile += 1
            # Update progress counter every 1000 pixels (reduced overhead)
            if progress_counter is not None and pixels_processed_in_tile % 1000 == 0:
                with progress_lock:
                    progress_counter.value += 1000

    # Update remaining pixels
    if progress_counter is not None:
        remaining = pixels_processed_in_tile % 1000
        if remaining > 0:
            with progress_lock:
                progress_counter.value += remaining

    # Save all pixels for this tile as a single .npy file
    # Shape: [num_pixels, num_months, 11]
    # Even if empty, save an empty array so resume check works
    npy_filename = muni_dir / f"{municipality_code}_tile_{tile_x}_{tile_y}.npy"

    if len(all_pixel_data) == 0:
        # Save empty array so resume check knows this tile was processed
        # Shape: [0, num_months, 11] - use float32 for compatibility
        tile_data = np.array([], dtype=np.float32).reshape(0, len(expected_months), 11)
        np.save(npy_filename, tile_data)
        return tile_data, 0, str(npy_filename)

    # Reconstruct array with optimized dtypes:
    # - Spectral bands: float32 (10 bands)
    # - DOY: uint8 (values 0-255) - 75% smaller than float32
    # Stack them together, but DOY will be stored as uint8 in memory
    # Note: NumPy arrays are homogeneous, so we'll use float32 for the final array
    # but we can optimize by ensuring DOY values fit in uint8 range
    num_months = len(expected_months)  # Use expected_months to ensure consistent shape
    num_pixels = len(all_pixel_data)

    # Pre-allocate array: [num_pixels, num_months, 11]
    tile_data = np.zeros((num_pixels, num_months, 11), dtype=np.float32)

    # Build final array (already efficient with pre-allocation)
    for i, (timeseries, doys) in enumerate(all_pixel_data):
        # Stack spectral bands (float32) and DOY (convert uint8 to float32 for array)
        tile_data[i, :, :10] = timeseries  # [num_months, 10]
        tile_data[i, :, 10] = doys.astype(
            np.float32
        )  # [num_months] - stored as float32 but values are 0-255

    np.save(npy_filename, tile_data)

    # Return pixel data instead of metadata (no index needed)
    return tile_data, len(all_pixel_data), str(npy_filename)


def process_municipality(municipality_code, tiff_files, output_dir, num_workers=None):
    """
    Process all .tiff files for one municipality.
    Extract all pixels from all tiles and save as a single .npy file per municipality.

    Args:
        municipality_code: Municipality code (e.g., '4100103')
        tiff_files: List of .tiff file paths for this municipality
        output_dir: Directory to save data

    Returns:
        Number of pixels processed
    """
    # Group files by month
    files_by_month = defaultdict(list)
    for tiff_file in tiff_files:
        _, year, month, _, _ = parse_filename(tiff_file)
        files_by_month[month].append(tiff_file)

    # Create output directory for this municipality
    muni_dir = output_dir / municipality_code
    muni_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique tile positions
    tile_positions = set()
    for tiff_file in tiff_files:
        _, _, _, tile_x, tile_y = parse_filename(tiff_file)
        tile_positions.add((tile_x, tile_y))

    # Determine expected months for this municipality (all tiles should have same months)
    expected_months = sorted(files_by_month.keys())
    num_expected_months = len(expected_months)

    print(
        f"Processing municipality {municipality_code}: {len(tile_positions)} tiles, {num_expected_months} months"
    )

    # Prepare tile info for processing
    tile_info = []

    for tile_x, tile_y in sorted(tile_positions):
        # Find all files for this tile across all months
        # Use expected_months to ensure all tiles have same month structure
        tile_files_by_month = {}
        for month in expected_months:
            for tiff_file in files_by_month[month]:
                _, _, file_month, file_tile_x, file_tile_y = parse_filename(tiff_file)
                if file_tile_x == tile_x and file_tile_y == tile_y:
                    tile_files_by_month[month] = tiff_file
                    break
            # If no file found for this month/tile, leave it out (process_tile will handle with zeros)

        if len(tile_files_by_month) > 0:
            # Get tile dimensions
            first_file = list(tile_files_by_month.values())[0]
            with rasterio.open(first_file) as src:
                height, width = src.height, src.width

            tile_info.append(
                {
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "tile_files_by_month": tile_files_by_month,
                    "estimated_pixels": height * width,
                }
            )

    # Calculate total expected pixels for progress tracking
    total_expected_pixels = sum(info["estimated_pixels"] for info in tile_info)

    # Create shared counter for progress tracking across workers
    manager = Manager()
    progress_counter = manager.Value("i", 0)
    progress_lock = manager.Lock()  # Lock for thread-safe updates
    active_tiles = manager.dict()  # Track which tiles are currently being processed

    # Prepare arguments for parallel processing
    # Pass expected_months to ensure all tiles produce arrays with same shape
    tile_args = [
        (
            municipality_code,
            info["tile_x"],
            info["tile_y"],
            info["tile_files_by_month"],
            output_dir,
            progress_counter,
            progress_lock,
            expected_months,  # Pass expected months to ensure consistent shapes
        )
        for info in tile_info
    ]

    # Process tiles in parallel
    if num_workers is None:
        num_workers = min(cpu_count(), len(tile_args))

    start_time = time.time()
    last_progress = manager.Value("i", 0)

    # Collect pixel data from all tiles to combine into single .npy file
    all_tile_data = []  # List of tile arrays
    completed_tiles_counter = manager.Value("i", 0)

    import threading

    def update_progress_bar():
        """Background thread to update progress bar from shared counter"""
        while completed_tiles_counter.value < len(tile_args):
            current_progress = progress_counter.value
            current_last = last_progress.value
            if current_progress > current_last:
                delta = current_progress - current_last
                tile_progress.update(delta)
                last_progress.value = current_progress
                elapsed = time.time() - start_time
                pixels_per_sec = current_progress / elapsed if elapsed > 0 else 0
                progress_pct = (
                    (current_progress / total_expected_pixels * 100)
                    if total_expected_pixels > 0
                    else 0
                )
                tile_progress.set_postfix(
                    {
                        "px": current_progress,
                        "px/s": f"{pixels_per_sec:.1f}",
                        "tiles": f"{completed_tiles_counter.value}/{len(tile_args)}",
                        "%": f"{progress_pct:.1f}%",
                    }
                )
            time.sleep(0.1)  # Update every 100ms

    with Pool(processes=num_workers) as pool:
        tile_progress = tqdm(
            total=total_expected_pixels,
            desc=f"  Tiles ({municipality_code})",
            leave=False,
            unit="px",
        )

        # Start background thread to update progress
        progress_thread = threading.Thread(target=update_progress_bar, daemon=True)
        progress_thread.start()

        # Process tiles and collect temporary file paths
        tile_files = []  # List of temporary tile .npy files
        for result in pool.imap_unordered(process_tile, tile_args):
            tile_data, num_pixels, temp_npy_path = result
            if temp_npy_path is not None and num_pixels > 0:
                tile_files.append(temp_npy_path)
            completed_tiles_counter.value += 1

        tile_progress.close()

    # Combine all tiles into single .npy file per municipality
    # Ensure output_dir is a Path object and resolve to absolute path for Windows compatibility
    output_dir = Path(output_dir).resolve()
    muni_dir = output_dir / municipality_code
    muni_dir.mkdir(parents=True, exist_ok=True)
    muni_npy_file = muni_dir / f"{municipality_code}.npy"

    # Track which tile files we successfully processed
    processed_tile_files = []

    try:
        if len(tile_files) > 0:
            # Load and concatenate all tiles
            tile_arrays = []
            for temp_tile_file in tile_files:
                tile_path = Path(temp_tile_file)
                if tile_path.exists():
                    try:
                        # Load normally (not memory-mapped) since we'll delete the file immediately
                        # Memory-mapping keeps file handle open, causing deletion issues on Windows
                        tile_data = np.load(tile_path)
                        if len(tile_data) > 0:
                            tile_arrays.append(tile_data)
                        processed_tile_files.append(tile_path)
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not load {tile_path.name}: {e}")
                        continue

            if len(tile_arrays) > 0:
                # Concatenate all tiles: [total_pixels, num_months, 11]
                combined_data = np.vstack(tile_arrays)
                # Ensure directory exists and use file handle for Windows Unicode path compatibility
                muni_npy_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Use file handle to avoid Windows Unicode path issues
                    with open(muni_npy_file, "wb") as f:
                        np.save(f, combined_data)
                except Exception as e:
                    print(
                        f"  ❌ Error saving combined data for {municipality_code}: {e}"
                    )
                    raise
                total_pixels = len(combined_data)
                print(f"  ✓ Saved {total_pixels:,} pixels to {muni_npy_file.name}")
            else:
                # All pixels were filtered out - skip saving empty file
                # Just return 0 to indicate no pixels were saved
                # This avoids creating empty files and path issues
                print(f"  ⚠️  Skipping {municipality_code} (all pixels filtered out)")
                total_pixels = 0
        else:
            # No tiles processed - skip saving empty file
            # Just return 0 to indicate no pixels were processed
            # This avoids creating empty files and path issues
            print(f"  ⚠️  Skipping {municipality_code} (no tiles to process)")
            total_pixels = 0

        # Clean up all temporary tile files (even if some failed to load)
        for tile_path in processed_tile_files:
            try:
                if tile_path.exists():
                    tile_path.unlink()
            except PermissionError:
                # On Windows, sometimes file deletion can fail if still in use
                # Try again after a brief delay
                time.sleep(0.1)
                try:
                    if tile_path.exists():
                        tile_path.unlink()
                except Exception:
                    # If still fails, just warn
                    print(
                        f"  ⚠️  Warning: Could not delete temporary file {tile_path.name}"
                    )
            except Exception as e:
                print(f"  ⚠️  Warning: Error deleting {tile_path.name}: {e}")

        # Also clean up any remaining tile files that weren't in processed_tile_files
        # (in case of errors during processing)
        for tile_file in muni_dir.glob(f"{municipality_code}_tile_*.npy"):
            if tile_file != muni_npy_file:  # Don't delete the final combined file
                try:
                    tile_file.unlink()
                except Exception:
                    pass  # Ignore errors when cleaning up

        return total_pixels

    except Exception as e:
        print(f"  ❌ Error processing {municipality_code}: {e}")
        # Try to clean up temporary files even on error
        for tile_path in processed_tile_files:
            try:
                if tile_path.exists():
                    tile_path.unlink()
            except Exception:
                pass
        raise


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("Fast Preprocessing: .tiff files → .npy files")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .tiff files (recursively in municipality subfolders)
    tiff_files = []
    for muni_folder in TIFF_ROOT_DIR.iterdir():
        if muni_folder.is_dir():
            tiff_files.extend(list(muni_folder.glob("*.tif")))

    print(f"Found {len(tiff_files)} .tiff files")

    # Group by municipality
    municipalities = defaultdict(list)
    for tiff_file in tiff_files:
        municipality_code, _, _, _, _ = parse_filename(tiff_file)
        municipalities[municipality_code].append(tiff_file)

    print(f"Found {len(municipalities)} municipalities")

    # Determine number of workers
    num_workers = min(cpu_count(), 12)  # Limit to avoid too many file handles
    print(f"Using {num_workers} parallel workers\n")

    # Process each municipality
    municipality_items = list(municipalities.items())
    muni_progress = tqdm(
        municipality_items, desc="Municipalities", unit="muni", position=0, leave=True
    )

    main_start_time = time.time()
    for municipality_code, muni_files in muni_progress:
        muni_progress.set_description(f"Municipality {municipality_code}")

        # Check if municipality is already processed (single .npy file exists)
        muni_dir = OUTPUT_DIR / municipality_code
        muni_npy_file = muni_dir / f"{municipality_code}.npy"

        if muni_npy_file.exists():
            # Check if file is non-empty
            try:
                data = np.load(muni_npy_file, mmap_mode="r")
                if len(data) > 0:
                    print(
                        f"  ⏭️  Skipping {municipality_code} (already processed - {len(data):,} pixels)"
                    )
                    # Clean up any leftover tile files
                    for tile_file in muni_dir.glob(f"{municipality_code}_tile_*.npy"):
                        try:
                            tile_file.unlink()
                        except Exception:
                            pass
                    continue
            except Exception:
                # File exists but might be corrupted, reprocess
                pass

        num_pixels = process_municipality(
            municipality_code, muni_files, OUTPUT_DIR, num_workers=num_workers
        )
        elapsed = time.time() - main_start_time
        total_pixels_processed = sum(
            len(np.load(OUTPUT_DIR / muni / f"{muni}.npy", mmap_mode="r"))
            for muni in municipalities.keys()
            if (OUTPUT_DIR / muni / f"{muni}.npy").exists()
        )
        pixels_per_sec = total_pixels_processed / elapsed if elapsed > 0 else 0
        muni_progress.set_postfix(
            {"pixels": total_pixels_processed, "px/s": f"{pixels_per_sec:.1f}"}
        )

    # Count total pixels processed
    total_pixels = 0
    municipalities_processed = 0
    for municipality_code in municipalities.keys():
        muni_npy_file = OUTPUT_DIR / municipality_code / f"{municipality_code}.npy"
        if muni_npy_file.exists():
            try:
                data = np.load(muni_npy_file, mmap_mode="r")
                total_pixels += len(data)
                municipalities_processed += 1
            except Exception:
                pass

    print(f"\n✓ Preprocessing complete!")
    print(f"  Processed {municipalities_processed} municipalities")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Saved as .npy files (one per municipality, no index needed)")

    # Create yield CSV template (using pandas for CSV compatibility)
    yield_template = pd.DataFrame(
        {
            "municipality_code": sorted(municipalities.keys()),
            "yield_tons": np.nan,  # Fill this with actual yield data
        }
    )
    yield_file = OUTPUT_DIR / "yield_template.csv"
    yield_template.to_csv(yield_file, index=False)
    print(f"  Yield template saved to: {yield_file}")
    print(f"  Please fill in yield_tons column with actual yield data")

    print(f"\n⚠️  Note: You'll need to adapt the dataset class to read from .npy files")
    print(f"    See: datasets/uscrops.py - modify __getitem__ to load from .npy")


if __name__ == "__main__":
    main()
    print(f"    See: datasets/uscrops.py - modify __getitem__ to load from .npy")


if __name__ == "__main__":
    main()
