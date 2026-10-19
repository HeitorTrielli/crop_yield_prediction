"""Build paper figures from Paraná year-LOO trial_005."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_arch_figure import main as build_arch_figure

ROOT = Path(__file__).resolve().parents[1]
LOO = ROOT / "loo_pixel_results/pr_only_trial_005"
MAMBORE = ROOT / "loo_pixel_results/intramunicipal_candidates"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = [2020, 2021, 2022, 2023, 2024]
MAIN_HEATMAP_YEARS = [2021, 2022]
APPENDIX_HEATMAP_YEARS = [2020, 2023, 2024]
TARGET_WIDTH_4K = 3840
COLORS = {
    2020: "#4c78a8",
    2021: "#54a24b",
    2022: "#e45756",
    2023: "#f58518",
    2024: "#b279a2",
}


def _save_mpl(fig: plt.Figure, stem: str, *, dpi: int = 300) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", stem)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _year_title(year: int) -> str:
    return f"{year} (drought LOO)" if year == 2022 else f"{year} (LOO)"


def write_image_panel(
    items: list[tuple[str, Path]],
    stem: str,
    *,
    nrows: int,
    ncols: int,
    target_width: int = TARGET_WIDTH_4K,
) -> None:
    """Compose source heatmaps at full resolution and export a 4K-wide PNG/PDF."""
    images: list[tuple[str, Image.Image]] = []
    for title, path in items:
        if not path.exists():
            raise FileNotFoundError(path)
        images.append((title, Image.open(path).convert("RGB")))

    cell_w = max(im.width for _, im in images)
    cell_h = max(im.height for _, im in images)
    title_h = max(48, cell_h // 18)
    pad = max(16, cell_w // 80)
    panel_w = ncols * cell_w + (ncols + 1) * pad
    panel_h = nrows * (cell_h + title_h) + (nrows + 1) * pad

    canvas = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(max(28, title_h // 2))

    for idx, (title, im) in enumerate(images):
        r, c = divmod(idx, ncols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + title_h + pad)
        draw.text((x0, y0), title, fill=(20, 20, 20), font=font)
        x = x0 + (cell_w - im.width) // 2
        y = y0 + title_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))

    if canvas.width != target_width:
        scale = target_width / canvas.width
        canvas = canvas.resize(
            (target_width, max(1, int(round(canvas.height * scale)))),
            Image.Resampling.LANCZOS,
        )

    png_path = OUT / f"{stem}.png"
    pdf_path = OUT / f"{stem}.pdf"
    canvas.save(png_path, format="PNG", optimize=True)
    canvas.save(pdf_path, format="PDF", resolution=300.0)
    print(f"wrote {stem} ({canvas.width}x{canvas.height})")


def load_scatter(year: int, path_kind: str) -> tuple[np.ndarray, np.ndarray]:
    path = (
        LOO
        / f"loo{year}/{path_kind}/figures/productivity_scatter"
        / f"productivity_{year}_direct.csv"
    )
    if not path.exists():
        path = LOO / f"loo{year}/{path_kind}/predictions/yield_forecasts_{year}.csv"
        pred, act = [], []
        with path.open() as f:
            for r in csv.DictReader(f):
                pred.append(float(r["forecast"]))
                act.append(float(r["actual_yield"]))
        return np.asarray(act), np.asarray(pred)
    act, pred = [], []
    with path.open() as f:
        for r in csv.DictReader(f):
            act.append(float(r["actual_productivity_t_ha"]))
            pred.append(float(r["predicted_productivity_t_ha"]))
    return np.asarray(act), np.asarray(pred)


def write_scatter_panel(path_kind: str, stem: str) -> None:
    lims = [0.5, 5.5]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    axes_flat = list(axes.ravel())
    for i, year in enumerate(YEARS):
        ax = axes_flat[i]
        act, pred = load_scatter(year, path_kind)
        ax.scatter(act, pred, s=8, alpha=0.55, c="#1f4e79", edgecolors="none")
        ax.plot(lims, lims, color="0.4", lw=0.8, ls="--")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_box_aspect(1)
        ax.set_title(_year_title(year), fontsize=9, pad=2)
        ax.tick_params(labelsize=7, length=2.5, pad=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel("Predicted (t/ha)", fontsize=8, labelpad=1)
        if i >= 3:
            ax.set_xlabel("Actual (t/ha)", fontsize=8, labelpad=1)
    axes_flat[-1].axis("off")
    fig.tight_layout()
    _save_mpl(fig, stem, dpi=300)


def load_incomplete_r2(path_kind: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for year in YEARS:
        path = (
            LOO
            / f"loo{year}/{path_kind}/predictions"
            / f"incomplete_series_evaluation_{year}.csv"
        )
        with path.open() as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["num_periods"]))
        out[year] = [float(r["r2"]) for r in rows]
    return out


def write_incomplete_r2(path_kind: str, stem: str) -> None:
    months = list(range(1, 7))
    r2_by_year = load_incomplete_r2(path_kind)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for year in YEARS:
        ys = r2_by_year[year]
        style = dict(lw=1.6 if year != 2022 else 1.8, marker="o", ms=4.5)
        label = f"{year} (drought)" if year == 2022 else str(year)
        ax.plot(months, ys, color=COLORS[year], label=label, **style)
    mean_excl = [
        float(np.mean([r2_by_year[y][i] for y in YEARS if y != 2022])) for i in range(6)
    ]
    ax.plot(
        months,
        mean_excl,
        color="0.15",
        lw=2.2,
        marker="s",
        ms=4.5,
        label="Mean excl. 2022",
        zorder=5,
    )
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xlim(0.8, 6.2)
    ax.set_xticks(months)
    ax.set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"])
    ax.set_xlabel("Last month included (season starts Oct)", fontsize=9)
    ax.set_ylabel(r"$R^2$", fontsize=9)
    ax.set_ylim(-1.05, 0.55)
    for year in YEARS:
        for k, r2 in enumerate(r2_by_year[year], start=1):
            if r2 < -1.05:
                ax.plot(k, -1.02, marker="v", ms=5, color=COLORS[year], linestyle="none")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    _save_mpl(fig, stem, dpi=300)


def _error_items(path_kind: str, years: list[int]) -> list[tuple[str, Path]]:
    return [
        (
            _year_title(y),
            LOO / f"loo{y}/{path_kind}/figures/municipal_heatmaps/map_pr_error_{y}.png",
        )
        for y in years
    ]


def _mambore_items(years: list[int]) -> list[tuple[str, Path]]:
    return [
        (
            _year_title(y),
            MAMBORE / f"mambore_{y}/mambore_pr_heatmap.png",
        )
        for y in years
    ]


def main() -> None:
    build_arch_figure()

    write_scatter_panel("pixel_by_pixel", "scatter_panel")
    write_scatter_panel("muni_mean", "scatter_panel_muni_mean")

    # Main text: best (2021) and worst (2022) years only, large 4K panels.
    write_image_panel(
        _error_items("pixel_by_pixel", MAIN_HEATMAP_YEARS),
        "map_error_panel",
        nrows=1,
        ncols=2,
    )
    write_image_panel(
        _mambore_items(MAIN_HEATMAP_YEARS),
        "mambore_panel",
        nrows=1,
        ncols=2,
    )
    # 2022 peak NDVI is less homogeneous than other LOO years, so spatial
    # structure is easier to compare with the productivity map.
    mambore_ndvi_year = 2022
    mambore_ndvi_dir = MAMBORE / f"mambore_{mambore_ndvi_year}"
    write_image_panel(
        [
            (
                f"Predicted productivity ({mambore_ndvi_year} LOO)",
                mambore_ndvi_dir / "mambore_pr_heatmap.png",
            ),
            ("Mean NDVI", mambore_ndvi_dir / "mambore_pr_ndvi_heatmap.png"),
            ("Peak NDVI", mambore_ndvi_dir / "mambore_pr_ndvi_peak_heatmap.png"),
        ],
        "mambore_ndvi_compare",
        nrows=3,
        ncols=1,
        target_width=3200,
    )

    # Appendix: remaining three LOO years + municipal-mean error maps.
    write_image_panel(
        _error_items("pixel_by_pixel", APPENDIX_HEATMAP_YEARS),
        "map_error_panel_appendix",
        nrows=1,
        ncols=3,
    )
    write_image_panel(
        _mambore_items(APPENDIX_HEATMAP_YEARS),
        "mambore_panel_appendix",
        nrows=1,
        ncols=3,
    )
    write_image_panel(
        _error_items("muni_mean", MAIN_HEATMAP_YEARS),
        "map_error_panel_muni_mean",
        nrows=1,
        ncols=2,
    )
    write_image_panel(
        _error_items("muni_mean", APPENDIX_HEATMAP_YEARS),
        "map_error_panel_muni_mean_appendix",
        nrows=1,
        ncols=3,
    )

    write_incomplete_r2("pixel_by_pixel", "incomplete_series_r2")
    write_incomplete_r2("muni_mean", "incomplete_series_r2_muni_mean")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
