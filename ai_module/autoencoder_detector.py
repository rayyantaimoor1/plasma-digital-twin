"""Sub-Module 2.2 (FE-2.2.6) - OPTIONAL autoencoder anomaly detector.

A second, DEEP-LEARNING anomaly detector alongside the Isolation Forest of
Sub-Module 2.2, enabling the documented classical-vs-deep-learning comparison
that FE-2.2.6 asks for, on the SAME physics-relationship-violation faults. A
lightweight PyTorch autoencoder is trained to reconstruct normal operating-
condition data; the reconstruction error (mean squared error) is the anomaly
score - a normal point reconstructs well (it lies on the learned normal
manifold), a relationship violation reconstructs poorly.

Key design contrast with the Isolation Forest detector (this IS the comparison):
the autoencoder is trained on the STANDARDISED RAW features, NOT on the
hand-engineered physics residuals. Sub-Module 2.2 showed Isolation Forest on raw
features fails on relationship violations (they sit inside the marginal hull of
every single feature), which is why it is fed residual summary features instead.
The autoencoder instead LEARNS the normal joint manifold directly: because it
learns, e.g., "at this pressure the electron temperature should be about X", an
input whose Te is inconsistent with its logged pressure cannot be reconstructed
well and produces a large error. Whether a neural network can therefore match the
classical detector WITHOUT the manual residual feature engineering is exactly the
question FE-2.2.6 poses; `compare_with_isolation_forest` reports the answer
honestly, whatever it is.

This is an OPTIONAL / stretch sub-module. PyTorch is scoped ONLY for this feature
(CLAUDE.md principle #7 and the FYP scope's Section 12); the core classification
task (2.1) deliberately stays tree-ensemble only. torch is therefore kept out of
the base requirements.txt (see requirements-optional.txt) and this module is the
only place in the project that imports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ai_module.anomaly_detection import (
    PlasmaAnomalyDetector,
    generate_normal_operating_data,
    normal_feature_ranges,
    range_check_is_anomaly,
)
from digital_twin.dataset_generation import FEATURE_COLUMNS

# Lightweight architecture (per FE-2.2.6's "lightweight autoencoder"): a small
# symmetric MLP with a compressive bottleneck. 7 features is a small input, so a
# 3-unit bottleneck forces the network to learn the low-dimensional manifold the
# normal data actually lives on (Te/n_e/etc. are all driven by just 2 inputs).
DEFAULT_HIDDEN = 8
DEFAULT_BOTTLENECK = 3
DEFAULT_EPOCHS = 400
DEFAULT_LEARNING_RATE = 1e-2
# Threshold at the 99th percentile of normal reconstruction error -> ~1% false
# positive rate by construction, matching the Isolation Forest detector's
# calibration so the two are compared on equal footing.
ANOMALY_QUANTILE = 0.99


class _Autoencoder(nn.Module):
    """Symmetric MLP autoencoder: features -> hidden -> bottleneck -> hidden ->
    features. ReLU in the hidden layers; the bottleneck and output layers are
    linear so the network can represent (and reconstruct) standardised features
    that take negative values."""

    def __init__(self, n_features: int, hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PlasmaAutoencoderDetector:
    """Deep-learning anomaly detector: reconstruction error of a PyTorch
    autoencoder trained on normal operating data. Higher score = more anomalous
    (the opposite sign convention to Isolation Forest's decision_function, which
    is negative for anomalies - documented here to avoid confusion)."""

    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        bottleneck: int = DEFAULT_BOTTLENECK,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[_Autoencoder] = None
        self.threshold: Optional[float] = None

    def _to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        scaled = self._scaler.transform(df[FEATURE_COLUMNS].to_numpy())
        return torch.from_numpy(scaled.astype("float32"))

    def fit(self, normal_df: pd.DataFrame) -> "PlasmaAutoencoderDetector":
        """Train the autoencoder to reconstruct normal data, then calibrate the
        anomaly threshold from the normal reconstruction-error distribution.

        A fixed seed makes CPU training deterministic (same weights, same
        threshold) so results are reproducible - important for a result that
        will be reported and defended."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._scaler = StandardScaler().fit(normal_df[FEATURE_COLUMNS].to_numpy())
        inputs = self._to_tensor(normal_df)

        self._model = _Autoencoder(len(FEATURE_COLUMNS), self.hidden, self.bottleneck)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            reconstruction = self._model(inputs)
            loss = loss_fn(reconstruction, inputs)
            loss.backward()
            optimizer.step()
        self._model.eval()

        normal_scores = self.reconstruction_error(normal_df)
        self.threshold = float(np.quantile(normal_scores, ANOMALY_QUANTILE))
        return self

    def reconstruction_error(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row mean squared reconstruction error in standardised feature
        space (the anomaly score; higher = more anomalous)."""
        if self._model is None:
            raise RuntimeError("Detector must be fit before scoring.")
        inputs = self._to_tensor(df)
        with torch.no_grad():
            reconstruction = self._model(inputs)
            return ((reconstruction - inputs) ** 2).mean(dim=1).numpy()

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.reconstruction_error(df)

    def is_anomaly(self, df: pd.DataFrame) -> np.ndarray:
        return self.reconstruction_error(df) > self.threshold


# ---------------------------------------------------------------------------
# Classical (Isolation Forest) vs deep-learning (autoencoder) comparison [FE-2.2.6]
# ---------------------------------------------------------------------------
@dataclass
class DetectorComparison:
    isolation_forest_recall: float
    isolation_forest_fpr: float
    autoencoder_recall: float
    autoencoder_fpr: float
    range_check_recall: float
    autoencoder_per_fault_recall: dict[str, float]
    n_normal: int
    n_anomalous: int


def compare_with_isolation_forest(
    normal_train: pd.DataFrame,
    normal_test: pd.DataFrame,
    anomalous_test: pd.DataFrame,
    seed: int = 0,
) -> DetectorComparison:
    """Fit both detectors on the same normal data and score them on the same
    injected physics-relationship violations, so the deep-learning method and the
    classical method are compared on identical footing [FE-2.2.6]. The naive
    range check is included as the shared baseline that both must beat."""
    iso = PlasmaAnomalyDetector(seed=seed).fit(normal_train)
    autoencoder = PlasmaAutoencoderDetector(seed=seed).fit(normal_train)

    anomalous_features = anomalous_test[FEATURE_COLUMNS]
    ranges = normal_feature_ranges(normal_test)

    ae_anomaly = autoencoder.is_anomaly(anomalous_features)
    per_fault = {
        fault: float(ae_anomaly[(anomalous_test["fault_type"] == fault).to_numpy()].mean())
        for fault in anomalous_test["fault_type"].unique()
    }
    return DetectorComparison(
        isolation_forest_recall=float(iso.is_anomaly(anomalous_features).mean()),
        isolation_forest_fpr=float(iso.is_anomaly(normal_test).mean()),
        autoencoder_recall=float(ae_anomaly.mean()),
        autoencoder_fpr=float(autoencoder.is_anomaly(normal_test).mean()),
        range_check_recall=float(range_check_is_anomaly(anomalous_features, ranges).mean()),
        autoencoder_per_fault_recall=per_fault,
        n_normal=len(normal_test),
        n_anomalous=len(anomalous_test),
    )


if __name__ == "__main__":
    # Run as a module:  python -m ai_module.autoencoder_detector
    from ai_module.anomaly_detection import generate_anomalous_data

    print("Training deep-learning (autoencoder) and classical (Isolation Forest) detectors...")
    normal_train = generate_normal_operating_data()
    normal_test = generate_normal_operating_data(seed=99, replicates=1)
    anomalous = generate_anomalous_data(n_samples=150)

    comparison = compare_with_isolation_forest(normal_train, normal_test, anomalous)
    print(f"\nNormal runs: {comparison.n_normal}   Anomalous runs: {comparison.n_anomalous}\n")
    print(f"  {'detector':<22} {'recall':>8} {'FPR':>8}")
    print(f"  {'Isolation Forest':<22} {comparison.isolation_forest_recall:>7.1%} {comparison.isolation_forest_fpr:>7.1%}")
    print(f"  {'Autoencoder (deep)':<22} {comparison.autoencoder_recall:>7.1%} {comparison.autoencoder_fpr:>7.1%}")
    print(f"  {'Range check (naive)':<22} {comparison.range_check_recall:>7.1%} {'-':>8}")
    print("\nAutoencoder recall by fault type:")
    for fault, recall in comparison.autoencoder_per_fault_recall.items():
        print(f"    {fault:26s} {recall:.1%}")
