"""Train ETA model versions and write artifacts plus a manifest.

Usage: python -m modelgate.model.train [--out artifacts] [--seed 7]

v1 is a 2-layer MLP trained for a short schedule. v2 is a 3-layer MLP trained
for a longer schedule with a lower final learning rate; it is the "candidate"
the serving layer promotes after a shadow run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from modelgate import __version__
from modelgate.model.data import (
    distance_baseline_mae,
    make_dataset,
    mean_absolute_error,
    naive_baseline_mae,
)
from modelgate.model.features import FEATURE_DIM, ZONE_IDS, feature_names
from modelgate.model.net import EtaNet, save_model

DEFAULT_SEED = 7
DATASET_SIZE = 12_000
TEST_FRACTION = 0.2


@dataclass(frozen=True)
class VersionSpec:
    version: str
    hidden: int
    depth: int
    epochs: int
    lr: float


VERSIONS: tuple[VersionSpec, ...] = (
    VersionSpec("v1", hidden=64, depth=2, epochs=10, lr=3e-3),
    VersionSpec("v2", hidden=96, depth=3, epochs=25, lr=2e-3),
)


def split(x: torch.Tensor, y: torch.Tensor, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(len(x), generator=g)
    n_test = int(len(x) * TEST_FRACTION)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def train_one(spec: VersionSpec, x_train: torch.Tensor, y_train: torch.Tensor, seed: int) -> EtaNet:
    torch.manual_seed(seed)
    model = EtaNet(hidden=spec.hidden, depth=spec.depth)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec.epochs)
    loss_fn = nn.SmoothL1Loss()
    batch = 256
    g = torch.Generator().manual_seed(seed + 2)
    model.train()
    for _ in range(spec.epochs):
        perm = torch.randperm(len(x_train), generator=g)
        for start in range(0, len(perm), batch):
            idx = perm[start : start + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def export(model: EtaNet, spec: VersionSpec, path: Path) -> None:
    save_model(model, path, hidden=spec.hidden, depth=spec.depth)


def train_all(out: Path, seed: int = DEFAULT_SEED, quiet: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    x, y, _ = make_dataset(DATASET_SIZE, seed)
    x_train, y_train, x_test, y_test = split(x, y, seed)
    baselines = {
        "mean_predictor_mae": round(naive_baseline_mae(y_train, y_test), 4),
        "distance_linear_mae": round(distance_baseline_mae(x_train, y_train, x_test, y_test), 4),
    }
    manifest: dict = {
        "modelgate_version": __version__,
        "seed": seed,
        "dataset": {"size": DATASET_SIZE, "test_fraction": TEST_FRACTION},
        "feature_schema": {
            "dim": FEATURE_DIM,
            "names": feature_names(),
            "zone_ids": list(ZONE_IDS),
            "inputs": {
                "distance_km": {"type": "float", "min": 0, "max": 500},
                "hour_of_day": {"type": "int", "min": 0, "max": 23},
                "day_of_week": {"type": "int", "min": 0, "max": 6},
                "pickup_zone_id": {"type": "int", "enum": list(ZONE_IDS)},
                "traffic_index": {"type": "float", "min": 0, "max": 1},
                "is_raining": {"type": "bool"},
            },
        },
        "baselines": baselines,
        "versions": {},
    }
    for spec in VERSIONS:
        t0 = time.perf_counter()
        model = train_one(spec, x_train, y_train, seed)
        with torch.no_grad():
            mae = mean_absolute_error(model(x_test), y_test)
        path = out / f"eta_{spec.version}.pt"
        export(model, spec, path)
        manifest["versions"][spec.version] = {
            "file": path.name,
            "arch": {"hidden": spec.hidden, "depth": spec.depth},
            "training": {"epochs": spec.epochs, "lr": spec.lr},
            "metrics": {"test_mae_minutes": round(mae, 4)},
            "train_seconds": round(time.perf_counter() - t0, 2),
        }
        if not quiet:
            print(
                f"{spec.version}: test MAE {mae:.3f} min "
                f"(mean baseline {baselines['mean_predictor_mae']:.3f}, "
                f"distance baseline {baselines['distance_linear_mae']:.3f}) "
                f"in {manifest['versions'][spec.version]['train_seconds']}s"
            )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = train_all(args.out, args.seed)
    print(f"wrote {args.out / 'manifest.json'} with {len(manifest['versions'])} versions")


if __name__ == "__main__":
    main()


__all__ = ["train_all", "VERSIONS", "VersionSpec"]
