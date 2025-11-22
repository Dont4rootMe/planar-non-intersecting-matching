from typing import Literal, Tuple, Union, Sequence
import argparse
import numpy as np

Normalize = Literal["none", "unit_square", "fit_square"]


def _to_unit_square(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    span[span == 0.0] = 1.0
    Z = (X - lo) / span
    return Z


def _fit_to_centered_square(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    c = (lo + hi) / 2.0
    half_span = (hi - lo) / 2.0
    s = float(max(half_span.max(), 1e-12))
    Z = (X - c) / s
    return np.clip(Z, -1.0, 1.0)


def _apply_norm(X: np.ndarray, normalize: Normalize) -> np.ndarray:
    if normalize == "none":
        return np.ascontiguousarray(X, dtype=np.float64)
    if normalize == "unit_square":
        return np.ascontiguousarray(_to_unit_square(X), dtype=np.float64)
    if normalize == "fit_square":
        return np.ascontiguousarray(_fit_to_centered_square(X), dtype=np.float64)
    raise ValueError("normalize must be one of {'none','unit_square','fit_square'}")


def get_nested_polygons(
    n_samples: int | Sequence[int] = 200,
    n_layers: int = 4,
    m_vertices: int | Sequence[int] = 24,
    r_min: float = 1.0,
    r_max: float = 4.0,
    noise: float = 0.03,
    sigma_tangent: float = 0.01,
    random_state: int | None = None,
    rotate_each_layer: bool = True,
    normalize: Normalize = "fit_square",
    return_labels: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Nested regular polygons with light noise. Perfect benchmark for onion decomposition.

    n_samples     — number of points (scalar or list of length n_layers).
    m_vertices    — number of vertices per layer (scalar or list of length n_layers).
    radii         — linear scale from r_min to r_max (outermost — r_max).
    noise         — δ_r~N(0,σ_r), δ_t~N(0,σ_t) (rad).
    """
    
    if noise <= 0.001:
        noise = 0.3
    noise = noise / 10.0
    
    sigma_radial = noise
    
    rng = np.random.default_rng(random_state)
    n_per_layer = n_samples // n_layers

    if isinstance(n_per_layer, int):
        nL = [n_per_layer] * n_layers
    else:
        nL = list(n_per_layer)
        assert len(nL) == n_layers
    if isinstance(m_vertices, int):
        mL = [m_vertices] * n_layers
    else:
        mL = list(m_vertices)
        assert len(mL) == n_layers

    # radii from outer to inner
    radii = np.linspace(r_max, r_min, n_layers)

    Xs = []
    ys = []
    for ℓ in range(n_layers):
        n = int(nL[ℓ])
        m = int(mL[ℓ])
        rℓ = float(radii[ℓ])
        φℓ = rng.uniform(0, 2*np.pi) if rotate_each_layer else 0.0

        # uniformly across vertices (additional uniform "shift" along arc)
        j = rng.integers(0, m, size=n)
        base_θ = (2*np.pi * j / m) + φℓ
        δt = rng.normal(0.0, sigma_tangent, size=n)
        δr = rng.normal(0.0, sigma_radial, size=n)

        θ = base_θ + δt
        r = rℓ + δr
        X = np.column_stack((r * np.cos(θ), r * np.sin(θ)))
        Xs.append(X)
        ys.append(np.full(n, ℓ, dtype=np.int32))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    X = _apply_norm(X, normalize)
    return (X, y) if return_labels else X


def get_pinwheel(
    n_samples: int = 1200,
    n_arms: int = 5,
    radius_loc: float = 1.0,
    radius_scale: float = 0.6,
    angle_gain: float = 0.9,
    angle_noise: float = 0.10,
    noise: float = 0.02,
    random_state: int | None = None,
    normalize: Normalize = "fit_square",
    return_labels: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Pinwheel: curved radial "rays". Nonlinearly separable, useful for hull stability testing.

    r ~ |N(radius_loc, radius_scale)|, θ = φ_k + angle_gain * r + ε, ε ~ N(0, angle_noise),
    x = r [cos θ, sin θ] + ξ, ξ ~ N(0, iso_noise^2 I).
    """
    if noise <= 0.001:
        noise = 0.2
    noise = noise / 20.0
    
    iso_noise = noise
    
    rng = np.random.default_rng(random_state)
    n_per = int(np.ceil(n_samples / n_arms))
    Xs = []
    ys = []
    for k in range(n_arms):
        φk = 2*np.pi * k / n_arms
        r = np.abs(rng.normal(radius_loc, radius_scale, size=n_per))
        ε = rng.normal(0.0, angle_noise, size=n_per)
        θ = φk + angle_gain * r + ε
        X = np.column_stack((r * np.cos(θ), r * np.sin(θ)))
        if iso_noise > 0:
            X += rng.normal(0.0, iso_noise, size=X.shape)
        Xs.append(X)
        ys.append(np.full(n_per, k, dtype=np.int32))
    X = np.vstack(Xs)[:n_samples]
    y = np.concatenate(ys)[:n_samples]
    X = _apply_norm(X, normalize)
    return (X, y) if return_labels else X


def get_spirals(
    n_samples: int = 400,
    n_arms: int = 2,
    turns: float = 1.75,
    a: float = 0.1,
    b: float = 0.25,
    angle_jitter: float = 0.02,
    noise: float = 0.02,
    random_state: int | None = None,
    normalize: Normalize = "fit_square",
    return_labels: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Archimedean spirals: r(θ)=a+bθ, θ ~ U[0, 2π·turns], for each ray offset φ_j=2π j/n_arms.

    angle_jitter — Gaussian deviation in angle (rad), noise — deviation in radius.
    """
    
    if noise <= 0.001:
        noise = 0.2
    
    noise = noise / 5.0
    
    rng = np.random.default_rng(random_state)
    
    n_per_arm = int(np.ceil(n_samples / n_arms))
    
    Xs = []
    ys = []
    for j in range(n_arms):
        θ = rng.uniform(0.0, 2*np.pi*turns, size=n_per_arm)
        θ += 2*np.pi * j / n_arms
        θ += rng.normal(0.0, angle_jitter, size=n_per_arm)
        r = a + b * θ + rng.normal(0.0, noise, size=n_per_arm)
        r = np.clip(r, 0.0, None)
        X = np.column_stack((r * np.cos(θ), r * np.sin(θ)))
        Xs.append(X)
        ys.append(np.full(n_per_arm, j, dtype=np.int32))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    X = _apply_norm(X, normalize)
    return (X, y) if return_labels else X