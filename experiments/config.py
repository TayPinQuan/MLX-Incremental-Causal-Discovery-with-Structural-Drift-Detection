"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


SCENARIOS = {
    'S1': dict(drift_type='structure',    drift_mode='abrupt'),
    'S2': dict(drift_type='coefficients', drift_mode='abrupt'),
    'S3': dict(drift_type='both',         drift_mode='abrupt'),
    'S4': dict(drift_type='structure',    drift_mode='gradual'),
    'S5': dict(drift_type='coefficients', drift_mode='gradual'),
}


@dataclass
class DataConfig:
    n_samples: int = 5000
    n_vars: int = 8
    max_lag: int = 2
    window_size: int = 500
    drift_at: int = 2500
    edge_prob: float = 0.2
    noise_std: float = 0.5


@dataclass
class ModelConfig:
    warmup: int = 3
    kappa: float = 5.0
    n_perm: int = 200
    z_threshold: float = 2.5
    use_mlx: bool = True
    ridge_alpha: float = 0.1
    coeff_tol: float = 0.1
    max_parents_per_node: int = 4
    residual_factor: float = 2.0
    coeff_explosion_factor: float = 3.0
    shd_absolute_threshold: float = 25.0
    max_local_fraction: float = 0.75
    min_block_edges: int = 3


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    scenarios: list[str] = field(
        default_factory=lambda: ['S1', 'S2', 'S3', 'S4', 'S5']
    )
    baselines: list[str] = field(
        default_factory=lambda: ['B1', 'B2', 'B3', 'B4']
    )
    out_dir: str = 'results'