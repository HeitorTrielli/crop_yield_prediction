"""
Script to create pixel-level prediction heatmaps for a municipality,
running STNetRegression directly on the original TIFF tiles.

Assumes TIFF filenames like:
    4100103_2023-01_tile_00_01.tif

and that each TIFF is multiband (10 bandas espectrais).
"""

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch.amp import autocast
from tqdm import tqdm

from datautils import getWeight
from STNetRegression import STNetRegression


# -------------------------------------------------------
# Util: mover estruturas para device
# -------------------------------------------------------
def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return type(x)(recursive_todevice(v, device) for v in x)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return x


# -------------------------------------------------------
# Parsing e transformação do pixel (igual ao seu .npy)
# -------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pixel-level prediction heatmaps from TIFFs."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_best.pth checkpoint file",
    )
    parser.add_argument(
        "--tiffpath",
        type=str,
        default="files/gee_images_30m",
        help="Path to directory containing original TIFF files (with municipality subfolders)",
    )
    parser.add_argument(
        "--municipality-code",
        type=str,
        required=True,
        help="Municipality code to create heatmap for (e.g., '4100103')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="heatmaps",
        help="Output directory for heatmap images",
    )
    parser.add_argument(
        "--sequencelength",
        type=int,
        default=6,
        help="Maximum length of time series data (default: 6)",
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="Whether to use random choice for time series data",
    )
    parser.add_argument(
        "--interp",
        action="store_true",
        help="Whether to interpolate the time series data",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Number of pixels to process at once (default: 2000)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available()',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Random seed for reproducibility (default: 27)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year of the data (default: 2023)",
    )

    args = parser.parse_args()

    args.tiffpath = Path(args.tiffpath)
    args.checkpoint = Path(args.checkpoint)
    args.output_dir = Path(args.output_dir)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def parse_filename_tiff(filename):
    """
    Parse TIFF filename.

    Espera algo como:
        4100103_2023-01_tile_00_01.tif

    Ajuste esta função se o seu padrão for diferente.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    # ['4100103', '2023-01', 'tile', '00', '01']
    municipality_code = parts[0]
    year_month = parts[1]
    year, month = year_month.split("-")
    tile_x = parts[3]
    tile_y = parts[4]
    return municipality_code, int(year), int(month), tile_x, tile_y


def month_midpoint_doy(year, month):
    """Dia do ano aproximado do meio do mês (15)."""
    dt = datetime(year, month, 15)
    return dt.timetuple().tm_yday


def transform_pixel(x, sequencelength, rc, interp, seed=None):
    """
    Transform pixel data: normalize, pad/sample to sequencelength, extract DOY.

    x: np.array de shape (T, 11) -> 10 bandas + DOY
    """
    if seed is not None:
        np.random.seed(seed)

    doy = x[:, -1].astype(np.int32)
    x = x[:, :10] * 1e-4  # mesmo scaling do seu código

    mean = np.array(
        [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]]
    )
    std = np.array([0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106])

    weight = getWeight(x)
    x = (x - mean) / std

    if interp:
        doy_pad = np.linspace(0, 366, sequencelength).astype("int")
        x_pad = np.array([np.interp(doy_pad, doy, x[:, i]) for i in range(10)]).T
        weight_pad = getWeight(x_pad * std + mean)
        mask = np.ones((sequencelength,), dtype=int)
    elif rc:
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()
        x_pad = x[idxs]
        mask = np.ones((sequencelength,), dtype=int)
        doy_pad = doy[idxs]
        weight_pad = weight[idxs]
        weight_pad /= weight_pad.sum()
    else:
        x_length, c_length = x.shape
        if x_length == sequencelength:
            mask = np.ones((sequencelength,), dtype=int)
            x_pad = x
            doy_pad = doy
            weight_pad = weight
            weight_pad /= weight_pad.sum()
        elif x_length < sequencelength:
            mask = np.zeros((sequencelength,), dtype=int)
            mask[:x_length] = 1
            x_pad = np.zeros((sequencelength, c_length))
            x_pad[:x_length, :] = x[:x_length, :]
            doy_pad = np.zeros((sequencelength,), dtype=int)
            doy_pad[:x_length] = doy[:x_length]
            weight_pad = np.zeros((sequencelength,), dtype=float)
            weight_pad[:x_length] = weight[:x_length]
            weight_pad /= weight_pad.sum() if weight_pad.sum() > 0 else 1
        else:
            idxs = np.random.choice(x.shape[0], sequencelength, replace=False)
            idxs.sort()
            x_pad = x[idxs]
            mask = np.ones((sequencelength,), dtype=int)
            doy_pad = doy[idxs]
            weight_pad = weight[idxs]
            weight_pad /= weight_pad.sum()

    return (
        torch.from_numpy(x_pad).type(torch.FloatTensor),
        torch.from_numpy(mask == 0),
        torch.from_numpy(doy_pad).type(torch.LongTensor),
        torch.from_numpy(weight_pad).type(torch.FloatTensor),
    )


# -------------------------------------------------------
# Leitura dos TIFFs e reconstrução espacial
# -------------------------------------------------------
def discover_tiles(tiff_dir, municipality_code, year):
    """
    Descobre a pasta correta mesmo quando o nome é '4100103_Abatiá' em vez de '4100103'.
    """

    # Encontra a pasta correta (prefixo = código)
    muni_dirs = list(tiff_dir.glob(f"{municipality_code}*"))
    if not muni_dirs:
        print(f"❌ Nenhuma pasta encontrada para {municipality_code} em {tiff_dir}")
        return None, None, None, None

    muni_dir = muni_dirs[0]   # pega a primeira que combina
    print(f"✔ Usando pasta: {muni_dir}")

    tiff_files = list(muni_dir.glob("*.tif"))
    if not tiff_files:
        print(f"❌ Não há TIFFs em {muni_dir}")
        return None, None, None, None

    tiles = {}
    for tiff_file in tiff_files:
        mcode, file_year, month, tile_x, tile_y = parse_filename_tiff(tiff_file.name)
        if mcode != municipality_code or file_year != year:
            continue
        key = (tile_x, tile_y)
        if key not in tiles:
            tiles[key] = {}
        tiles[key][month] = tiff_file

    if not tiles:
        print(f"❌ Nenhum TIFF do ano {year} encontrado para {municipality_code}")
        return None, None, None, None

    # usa o primeiro arquivo para pegar shape
    first_tile = next(iter(tiles.values()))
    first_file = next(iter(first_tile.values()))
    with rasterio.open(first_file) as src:
        tile_height, tile_width = src.height, src.width
        num_bands = src.count

    return tiles, tile_height, tile_width, num_bands


def run_model_on_tiffs(
    model,
    municipality_code,
    tiffpath,
    year,
    sequencelength,
    rc,
    interp,
    chunk_size,
    device,
    seed=None,
):
    """
    Lê todos os TIFFs do município/ano, monta as séries temporais por pixel,
    roda o modelo em batches e devolve uma matriz 2D com as previsões no layout
    original dos tiles, alinhado pelas coordenadas geográficas.
    """
    tiles, tile_height, tile_width, num_bands = discover_tiles(
        tiffpath, municipality_code, year
    )
    if tiles is None:
        return None

    # ------------------------------------------------------------------
    # 1) Descobrir posição (x, y) de cada tile usando o transform do TIFF
    #    Vamos usar o canto superior esquerdo como referência
    # ------------------------------------------------------------------
    tile_positions = {}  # (tile_x, tile_y) -> (x0, y0)
    for tile_key, month_dict in tiles.items():
        # pega qualquer mês desse tile (o primeiro da lista)
        some_month = sorted(month_dict.keys())[0]
        some_file = month_dict[some_month]
        with rasterio.open(some_file) as src:
            # coordenadas do pixel (0, 0)
            x0, y0 = src.transform * (0, 0)
        tile_positions[tile_key] = (x0, y0)

    # lista de coordenadas únicas em X (oeste-leste) e Y (norte-sul)
    x_vals = sorted({pos[0] for pos in tile_positions.values()})          # esquerda -> direita
    y_vals = sorted({pos[1] for pos in tile_positions.values()}, reverse=True)  # topo -> baixo

    # mapeia tile -> índice da linha/coluna na grade de tiles
    tile_to_col = {
        tile_key: x_vals.index(tile_positions[tile_key][0]) for tile_key in tiles
    }
    tile_to_row = {
        tile_key: y_vals.index(tile_positions[tile_key][1]) for tile_key in tiles
    }

    # ------------------------------------------------------------------
    # 2) Tamanho do mosaico final
    # ------------------------------------------------------------------
    n_rows_tiles = len(y_vals)
    n_cols_tiles = len(x_vals)

    total_height = n_rows_tiles * tile_height
    total_width = n_cols_tiles * tile_width

    heatmap = np.full((total_height, total_width), np.nan, dtype=np.float32)

    model.eval()

    # ------------------------------------------------------------------
    # 3) Loop sobre tiles, montar séries temporais e preencher o mosaico
    # ------------------------------------------------------------------
    for (tile_x, tile_y), month_dict in tqdm(
        tiles.items(), desc="Processing tiles", unit="tile"
    ):
        months = sorted(month_dict.keys())
        num_times = len(months)

        # Carrega todos os meses deste tile em memória
        tile_stack = None
        for t_idx, month in enumerate(months):
            tiff_file = month_dict[month]
            with rasterio.open(tiff_file) as src:
                data = src.read()  # (bands, H, W)
                data = data.astype(np.float32)

                if tile_stack is None:
                    tile_stack = np.zeros(
                        (num_times, src.count, src.height, src.width),
                        dtype=np.float32,
                    )
                tile_stack[t_idx] = data

        # DOYs para cada time step
        doys = np.array([month_midpoint_doy(year, m) for m in months], dtype=np.float32)

        # índice da "linha" e "coluna" de tiles no mosaico global
        tile_row_idx = tile_to_row[(tile_x, tile_y)]
        tile_col_idx = tile_to_col[(tile_x, tile_y)]

        current_chunk = []
        chunk_positions = []

        # Usa a altura/largura reais deste tile (podem ser menores na borda)
        local_height = tile_stack.shape[2]
        local_width = tile_stack.shape[3]

        # Loop sobre pixels do tile
        for row in range(local_height):
            for col in range(local_width):
                # vetor (T, bands)
                spectral_ts = tile_stack[:, :, row, col]  # (T, num_bands)

                # monta x: (T, 11) = 10 bandas + DOY
                x = np.zeros((num_times, 11), dtype=np.float32)
                x[:, :10] = spectral_ts[:, :10]
                x[:, -1] = doys

                X_tuple = transform_pixel(
                    x, sequencelength, rc=rc, interp=interp, seed=seed
                )
                current_chunk.append(X_tuple)

                # posição global na matriz grande
                global_row = tile_row_idx * tile_height + row
                global_col = tile_col_idx * tile_width + col
                chunk_positions.append((global_row, global_col))

                if len(current_chunk) >= chunk_size:
                    process_chunk(
                        model,
                        current_chunk,
                        chunk_positions,
                        heatmap,
                        device,
                    )
                    current_chunk = []
                    chunk_positions = []

        # processa resto do tile
        if current_chunk:
            process_chunk(
                model,
                current_chunk,
                chunk_positions,
                heatmap,
                device,
            )

    return heatmap




def process_chunk(model, chunk, positions, heatmap, device):
    """Roda o modelo num batch de pixels e escreve no heatmap."""
    chunk_x = torch.stack([p[0] for p in chunk])
    chunk_mask = torch.stack([p[1] for p in chunk])
    chunk_doy = torch.stack([p[2] for p in chunk])
    chunk_weight = torch.stack([p[3] for p in chunk])

    X_chunk = (chunk_x, chunk_mask, chunk_doy, chunk_weight)
    X_chunk = recursive_todevice(X_chunk, device)

    with (
        autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.no_grad()
    ):
        preds = model(X_chunk).detach().cpu().numpy().reshape(-1)

    H, W = heatmap.shape
    for (row, col), pred in zip(positions, preds):
        # garante que não estoura os limites do heatmap
        if 0 <= row < H and 0 <= col < W:
            heatmap[row, col] = pred



# -------------------------------------------------------
# Visualização
# -------------------------------------------------------
def create_heatmap_figure(heatmap, output_path, municipality_code):
    valid = ~np.isnan(heatmap)
    if not valid.any():
        print("  ⚠️  No valid predictions to visualize.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="lightgray", alpha=0.3)

    im = ax.imshow(heatmap, cmap=cmap, interpolation="nearest", aspect="equal")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Yield Prediction (tons)", rotation=270, labelpad=20)

    ax.set_title(
        f"Yield Prediction Heatmap - Municipality {municipality_code}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    vals = heatmap[valid]
    stats_text = (
        f"Min: {vals.min():.2f} tons\n"
        f"Max: {vals.max():.2f} tons\n"
        f"Mean: {vals.mean():.2f} tons\n"
        f"Total (sum over pixels): {vals.sum():.2f} tons\n"
        f"Pixels: {vals.size:,}"
    )
    
    fig.text(
    0.5,
    0.02,
    stats_text,
    ha="center",
    va="bottom",
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
)


    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved heatmap to {output_path}")


# -------------------------------------------------------
# main
# -------------------------------------------------------
def main():
    args = parse_args()

    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print("Creating model...")
    model = STNetRegression(
        input_dim=10,
        num_outputs=1,
        max_seq_len=args.sequencelength,
    ).to(device)

    state_dict = checkpoint["model_state"]
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nGenerating pixel-level predictions from TIFFs for municipality {args.municipality_code}..."
    )
    heatmap = run_model_on_tiffs(
        model,
        args.municipality_code,
        args.tiffpath,
        args.year,
        args.sequencelength,
        args.rc,
        args.interp,
        args.chunk_size,
        device,
        seed=args.seed,
    )

    if heatmap is None:
        print("  ❌ Failed to generate predictions from TIFFs.")
        return

    output_path = args.output_dir / f"{args.municipality_code}_tiff_heatmap.png"
    print("\nCreating heatmap figure...")
    create_heatmap_figure(heatmap, output_path, args.municipality_code)

    print(f"\n✓ Done! Heatmap saved to {output_path}")


if __name__ == "__main__":
    main()
