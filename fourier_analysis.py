from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable
import numpy as np
import torch 
import matplotlib.pyplot as plt

def sample_height_grid(height_model, n):
    axis = torch.linspace(0.0, 1.0, n, dtype=torch.double)
    xx, zz = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )
    xz = torch.stack([xx.reshape(-1), zz.reshape(-1)], dim=1)

    with torch.no_grad():
        heights = height_model(xz).reshape(n, n)

    return xz, heights

def remove_best_fit_plane(xz: torch.Tensor, heights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove ax+bz+c so the spectrum focuses on shape rather than global tilt."""
    design = torch.cat([xz, torch.ones((xz.shape[0], 1), dtype=xz.dtype, device=xz.device)], dim=1)
    coefficients = torch.linalg.lstsq(design, heights.reshape(-1)).solution
    plane = (design @ coefficients).reshape_as(heights)
    return heights - plane, coefficients


def radial_average(normalized_radius: torch.Tensor, power: torch.Tensor, bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the 2D power spectrum inside concentric frequency rings."""
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=power.dtype, device=power.device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = torch.zeros_like(centers)

    for index in range(bins):
        if index == bins - 1:
            mask = (normalized_radius >= edges[index]) & (normalized_radius <= edges[index + 1])
        else:
            mask = (normalized_radius >= edges[index]) & (normalized_radius < edges[index + 1])
        if mask.any():
            values[index] = power[mask].mean()

    return centers, values

def analyze_height_spectrum(
    mlp,
    n=128,
    cutoff=0.25,
    out="results/geodesics/fourier_analysis.png",
    show=True,
):
    # 1. Sample the continuous MLP on an n x n grid.
    parameter = next(mlp.parameters())
    axis = torch.linspace(
        0.0,
        1.0,
        n,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    xx, zz = torch.meshgrid(axis, axis, indexing="ij")

    xz = torch.stack([xx.reshape(-1), zz.reshape(-1)], dim=1)

    with torch.no_grad():
        heights = mlp(xz).reshape(n, n)

    # 2. Remove the best-fit plane ax + bz + c.
    design = torch.cat([xz, torch.ones((xz.shape[0], 1), dtype=xz.dtype, device=xz.device)], dim=1)

    plane_coefficients = torch.linalg.lstsq(design, heights.reshape(-1)).solution

    plane = (design @ plane_coefficients).reshape(n, n)
    residual = heights - plane

    # 3. Apply a Hann window to reduce boundary artifacts.
    window_1d = torch.hann_window(n, periodic=False, dtype=heights.dtype, device=heights.device)
    window = window_1d[:, None] * window_1d[None, :]
    windowed_residual = residual * window

    # 4. Compute the 2D Fourier coefficients and power.
    coefficients = torch.fft.fftshift(
        torch.fft.fft2(
            windowed_residual,
            norm="ortho",
        )
    )
    power = coefficients.abs().square()

    # 5. Find the radial frequency of every coefficient.
    spacing = 1.0 / (n - 1)
    frequencies = torch.fft.fftshift(
        torch.fft.fftfreq(
            n,
            d=spacing,
            dtype=heights.dtype,
            device=heights.device,
        )
    )
    frequency_x, frequency_z = torch.meshgrid(frequencies, frequencies, indexing="ij")
    radius = torch.sqrt(frequency_x.square() + frequency_z.square())
    normalized_radius = radius / radius.max().clamp_min(1e-12)

    # 6. Measure low-frequency and high-frequency energy.
    low_mask = (radius > 0) & (normalized_radius <= cutoff)
    high_mask = normalized_radius > cutoff

    low_energy = power[low_mask].sum()
    high_energy = power[high_mask].sum()
    total_energy = (low_energy + high_energy).clamp_min(1e-30)

    low_ratio = (low_energy / total_energy).item()
    high_ratio = (high_energy / total_energy).item()

    # 7. Average the power inside radial frequency rings.
    bin_edges = torch.linspace(
        0.0,
        1.0,
        41,
        dtype=power.dtype,
        device=power.device,
    )
    radial_frequency = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    radial_power = []
    for index in range(len(bin_edges) - 1):
        mask = (
            (normalized_radius >= bin_edges[index])
            & (normalized_radius < bin_edges[index + 1])
        )

        if mask.any():
            radial_power.append(power[mask].mean())
        else:
            radial_power.append(
                torch.tensor(
                    float("nan"),
                    dtype=power.dtype,
                    device=power.device,
                )
            )

    radial_power = torch.stack(radial_power)

    # 8. Plot the 2D and radial power spectra.
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    log_power = np.log10(power.detach().cpu().numpy() + 1e-20)

    extent = (
        frequencies[0].item(),
        frequencies[-1].item(),
        frequencies[0].item(),
        frequencies[-1].item(),
    )

    spectrum_image = axes[0].imshow(
        log_power,
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
    )
    axes[0].set_title("2D log power spectrum")
    axes[0].set_xlabel("frequency z")
    axes[0].set_ylabel("frequency x")
    figure.colorbar(
        spectrum_image,
        ax=axes[0],
        label="log10 power",
    )

    axes[1].semilogy(
        radial_frequency.detach().cpu().numpy(),
        radial_power.detach().cpu().numpy() + 1e-20,
    )
    axes[1].axvline(
        cutoff,
        color="red",
        linestyle="--",
        label=f"cutoff={cutoff}",
    )
    axes[1].set_title("Radial power spectrum")
    axes[1].set_xlabel("normalized radial frequency")
    axes[1].set_ylabel("mean power")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle(
        f"Low frequency: {low_ratio:.2%}   "
        f"High frequency: {high_ratio:.2%}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)

    if show:
        plt.show()

    plt.close(figure)

    print(
        f"Fourier high-frequency energy: {high_ratio:.2%}; "
        f"saved to {output_path.resolve()}",
        flush=True,
    )

    return high_ratio