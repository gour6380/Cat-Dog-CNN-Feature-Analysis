"""Identity-matched, resumable computation of independent CNN feature families."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from src.config import ExperimentConfig
from src.data import (
    PetRecordDataset,
    SampleRecord,
    load_registered_splits,
    stratified_hash_selection,
)
from src.feature_section_types import FEATURE_SECTIONS, FeatureSection, SectionData
from src.io_utils import (
    atomic_save_npz,
    atomic_write_json,
    canonical_json_bytes,
    environment_snapshot,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from src.model import build_model, load_checkpoint_model, state_dict_hash
from src.progress import status
from src.runtime import memory_snapshot, synchronize
from src.training import Arm, checkpoint_path, experiment_provenance

ARMS: tuple[Arm, ...] = ("standard", "adversarial")
METHOD_FILES = (
    "src/feature_visualization.py",
    "src/feature_sections.py",
    "src/feature_section_types.py",
    "src/feature_pattern_sections.py",
    "src/feature_anchor_sections.py",
    "src/feature_methods.py",
    "src/stage_walkthrough.py",
    "src/classifier_diagnostics.py",
    "src/config.py",
    "src/data.py",
    "src/model.py",
    "src/attacks.py",
    "src/io_utils.py",
    "src/runtime.py",
    "requirements.txt",
)


class FeatureEvidenceError(ValueError):
    """Saved features cannot be attributed to the current registered analysis."""


@contextmanager
def _preserved_randomness(device: torch.device) -> Iterator[None]:
    """Model construction must not reseed the surrounding notebook's RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise FeatureEvidenceError(f"feature receipt is not an object: {path}")
    return value


def _arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = {key: archive[key].copy() for key in archive.files}
    except (ValueError, OSError) as error:
        raise FeatureEvidenceError(f"unreadable feature measurements: {path}") from error
    if any(
        np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()
        for value in result.values()
    ):
        raise FeatureEvidenceError("nonfinite feature measurements")
    return result


def _inside(config: ExperimentConfig, relative: str, digest: str) -> Path:
    path = (config.root / relative).resolve()
    if not path.is_relative_to(config.root) or not path.is_file() or sha256_file(path) != digest:
        raise FeatureEvidenceError(f"missing/stale feature artifact: {relative}")
    return path


class FeatureWorkspace:
    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device,
        progress: bool,
        *,
        read_only: bool = False,
    ) -> None:
        from src.feature_visualization import LAYERS

        self.config, self.device, self.progress = config, device, progress
        if config.label_mode != "species":
            raise FeatureEvidenceError("feature sections require the cat/dog head")
        values = config.section("feature_visualization")
        if values.get("layers") != LAYERS:
            raise FeatureEvidenceError("feature levels must be stem ReLU, layer2 and layer4")
        size, tile = config.integer("input", "size"), int(values["occlusion_patch_size"])
        if tile <= 0 or size % tile or int(values["occlusion_stride"]) != tile:
            raise FeatureEvidenceError("occlusion requires a nonoverlapping equal-area tile grid")
        splits = load_registered_splits(config)
        seed = int(values["seed"])
        self.references = stratified_hash_selection(
            splits["calibration"],
            int(values["calibration_per_class"]),
            f"features:{seed}:calibration",
        )
        self.anchors = stratified_hash_selection(
            splits["test"],
            int(values["anchors_per_class"]),
            f"features:{seed}:anchors",
        )
        self.checkpoints = {arm: sha256_file(checkpoint_path(config, arm)) for arm in ARMS}
        if read_only:
            from src.training_monitoring import training_monitoring_enabled

            if training_monitoring_enabled(config):
                for name in ("monitoring-protocol.json", "monitoring-source.json"):
                    path = config.project_path("artifacts") / "training" / name
                    if not path.is_file():
                        raise FeatureEvidenceError(
                            "registered training identity is missing; "
                            "read-only display cannot create it"
                        )
        self.provenance = experiment_provenance(config)
        self.method_sources = {name: sha256_file(config.root / name) for name in METHOD_FILES}
        self.method_sha256 = sha256_bytes(canonical_json_bytes(self.method_sources))
        self.selection = {
            "calibration_sample_ids": [r.sample_id for r in self.references],
            "test_anchor_ids": [r.sample_id for r in self.anchors],
            "channel_rule": "largest balanced-calibration mean peak responses; tie by channel ID",
            "diagnostic_channels_per_layer": int(values["channels_per_layer"]),
            "walkthrough_channels_per_stage": 3,
            "photograph_sha256": {
                record.sample_id: sha256_file(Path(record.image_path))
                for record in [*self.references, *self.anchors]
            },
        }
        self.identity = sha256_bytes(
            canonical_json_bytes(
                {
                    "config": config.sha256,
                    "checkpoints": self.checkpoints,
                    "provenance": self.provenance,
                    "method": self.method_sha256,
                    "selection": self.selection,
                    "device": str(device),
                }
            )
        )
        self.cache_root = config.project_path("artifacts") / "features" / "sections"

    def _path(self, arm: str, section: str) -> Path:
        return self.cache_root / self.identity[:16] / arm / f"{section}.json"

    def _envelope(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "__reference_sample_ids": np.asarray([r.sample_id for r in self.references]),
            "__anchor_sample_ids": np.asarray([r.sample_id for r in self.anchors]),
        }

    def _load(
        self,
        arm: str,
        section: str,
        *,
        _visiting: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[dict[str, Any], SectionData] | None:
        cache_key = (arm, section)
        if cache_key in _visiting:
            return None
        path = self._path(arm, section)
        if not path.is_file():
            return None
        try:
            receipt = _read(path)
            if (
                receipt.get("identity") != self.identity
                or receipt.get("arm") != arm
                or receipt.get("section") != section
            ):
                return None
            arrays = _arrays(_inside(self.config, receipt["arrays_path"], receipt["arrays_sha256"]))
            for key, expected in self._envelope().items():
                if key not in arrays or not np.array_equal(arrays[key], expected):
                    return None
            if "anchor_sample_ids" in arrays and not np.array_equal(
                arrays["anchor_sample_ids"], self._envelope()["__anchor_sample_ids"]
            ):
                return None
            for figure in receipt.get("figures", []):
                if figure.get("identity") != self.identity or figure.get("section") != section:
                    return None
                _inside(self.config, figure["path"], figure["sha256"])
            for prerequisite in receipt.get("dependencies", []):
                dependency_path = _inside(self.config, prerequisite["path"], prerequisite["sha256"])
                dependency = _read(dependency_path)
                dependency_arm, dependency_section = dependency["arm"], dependency["section"]
                if (
                    not isinstance(dependency_arm, str)
                    or not isinstance(dependency_section, str)
                    or dependency_path != self._path(dependency_arm, dependency_section).resolve()
                    or self._load(
                        dependency_arm, dependency_section, _visiting=_visiting | {cache_key}
                    )
                    is None
                ):
                    return None
            if (
                arm in ARMS
                and section in FEATURE_SECTIONS
                and receipt.get("payload", {}).get("state_unchanged") is not True
            ):
                return None
            return receipt, SectionData(receipt["payload"], arrays)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save(
        self,
        arm: str,
        section: str,
        data: SectionData,
        *,
        figures: list[dict[str, Any]] | None = None,
        dependencies: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        path = self._path(arm, section)
        arrays = {**data.arrays, **self._envelope()}
        if any(
            np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()
            for value in arrays.values()
        ):
            raise FeatureEvidenceError("nonfinite feature measurements")
        atomic_save_npz(path.with_suffix(".npz"), arrays)
        receipt = {
            "schema_version": 2,
            "created_at": utc_now(),
            "identity": self.identity,
            "config_sha256": self.config.sha256,
            "method_sha256": self.method_sha256,
            "arm": arm,
            "section": section,
            "device": str(self.device),
            "arrays_path": str(path.with_suffix(".npz").relative_to(self.config.root)),
            "arrays_sha256": sha256_file(path.with_suffix(".npz")),
            "payload": data.payload,
            "figures": figures or [],
            "dependencies": dependencies or [],
        }
        atomic_write_json(path, receipt)
        return receipt

    def _pixels(self, records: list[SampleRecord]) -> torch.Tensor:
        dataset = PetRecordDataset(records, self.config, training=False)
        return (
            torch.stack([dataset[index][0] for index in range(len(records))])
            .contiguous()
            .to(self.device)
        )

    def prepare_initial(self, *, baseline: bool) -> None:
        from src.feature_visualization import LAYERS, _np, _probe

        need_kernels = self._load("initial", "kernels") is None
        need_baseline = baseline and self._load("initial", "probe") is None
        if not need_kernels and not need_baseline:
            return
        initial = build_model(self.config)
        try:
            if need_kernels:
                convolution = initial.get_submodule("network.conv1")
                if not isinstance(convolution, nn.Conv2d):
                    raise FeatureEvidenceError("initialization has no ResNet stem convolution")
                self._save(
                    "initial",
                    "kernels",
                    SectionData(arrays={"kernels": _np(convolution.weight)}),
                )
            if need_baseline:
                initial.to(self.device).eval()
                probes = _probe(
                    initial,
                    self._pixels(self.references),
                    self.config.integer("training", "evaluation_batch_size"),
                )
                self._save(
                    "initial",
                    "probe",
                    SectionData(
                        arrays={
                            f"{index}_{name}": _np(probes[layer][name])
                            for index, layer in enumerate(LAYERS)
                            for name in ("means", "peaks", "positions")
                        }
                    ),
                )
        finally:
            initial.cpu()
            del initial
            if self.device.type == "mps":
                torch.mps.empty_cache()

    def run(self, arm: Arm, section: FeatureSection) -> bool:
        if self._load(arm, section) is not None:
            return False
        if section == "diagnostics":
            for dependency in ("gradcam", "occlusion"):
                self.run(arm, dependency)
                self.manifest()
        if section in {"kernels", "diagnostics"}:
            self.prepare_initial(baseline=section == "diagnostics")
        from src.feature_anchor_sections import compute_anchors
        from src.feature_pattern_sections import compute_patterns

        status(f"Features: {arm} · {section}", enabled=self.progress)
        started = time.perf_counter()
        model = load_checkpoint_model(self.config, checkpoint_path(self.config, arm), self.device)
        before = state_dict_hash(model.state_dict())
        modes = [(module, module.training) for module in model.modules()]
        gradients = [
            (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
            for parameter in model.parameters()
        ]
        context = SectionContext(self, arm, section, model)
        model.eval()
        try:
            synchronize(self.device)
            data = (
                compute_patterns(section, context)
                if section in {"kernels", "synthetic", "real_patches"}
                else compute_anchors(section, context)
            )
            synchronize(self.device)
            if state_dict_hash(model.state_dict()) != before:
                raise FeatureEvidenceError(
                    "feature analysis changed model weights or BatchNorm state"
                )
            data.payload["state_unchanged"] = True
            data.payload["elapsed_seconds"] = time.perf_counter() - started
            self._save(
                arm, section, data, figures=context.figures, dependencies=context.dependencies
            )
        finally:
            for module, training in modes:
                module.training = training
            for parameter, gradient in gradients:
                parameter.grad = gradient
            model.cpu()
            del context, model
            if self.device.type == "mps":
                torch.mps.empty_cache()
        return True

    def manifest(self) -> dict[str, Any]:
        from src.feature_visualization import LIMITATIONS

        arms: dict[str, Any] = {}
        figures: list[dict[str, Any]] = []
        receipts: dict[str, Any] = {}
        missing: list[str] = []
        elapsed_seconds = 0.0
        for arm in ARMS:
            combined: dict[str, Any] = {
                "checkpoint_sha256": self.checkpoints[arm],
                "layers": {},
                "anchors": [],
                "stage_walkthroughs": [],
            }
            anchor_data: dict[str, dict[str, Any]] = {}
            receipts[arm] = {}
            for section in FEATURE_SECTIONS:
                cached = self._load(arm, section)
                if cached is None:
                    missing.append(f"{arm}/{section}")
                    continue
                receipt, data = cached
                elapsed_seconds += float(data.payload.get("elapsed_seconds", 0.0))
                path = self._path(arm, section)
                receipts[arm][section] = {
                    "path": str(path.relative_to(self.config.root)),
                    "sha256": sha256_file(path),
                }
                figures.extend(receipt["figures"])
                for layer, values in data.payload.get("layers", {}).items():
                    combined["layers"].setdefault(layer, {}).update(values)
                for anchor in data.payload.get("anchors", []):
                    anchor_data.setdefault(anchor["sample_id"], {}).update(anchor)
                combined["stage_walkthroughs"].extend(data.payload.get("stage_walkthroughs", []))
            combined["anchors"] = [
                anchor_data[record.sample_id]
                for record in self.anchors
                if record.sample_id in anchor_data
            ]
            combined["state_unchanged"] = not any(item.startswith(arm + "/") for item in missing)
            arms[arm] = combined
        result = {
            "schema_version": 2,
            "created_at": utc_now(),
            "identity": self.identity,
            "config_sha256": self.config.sha256,
            "method_sha256": self.method_sha256,
            "method_source_files_sha256": self.method_sources,
            "device": str(self.device),
            "provenance": self.provenance,
            "selection": self.selection,
            "arms": arms,
            "figures": figures,
            "section_receipts": receipts,
            "limitations": LIMITATIONS,
            "status": "partial" if missing else "complete",
            "completion": {"required_sections": list(FEATURE_SECTIONS), "missing": missing},
            "elapsed_seconds": elapsed_seconds,
            "timing_scope": "sum of recorded section computation; excludes cache hydration "
            "and shared initial preparation",
            "environment": environment_snapshot(),
            "final_memory": memory_snapshot(self.device),
        }
        atomic_write_json(
            self.config.project_path("results") / "feature_visualizations.partial.json", result
        )
        if not missing:
            atomic_write_json(
                self.config.project_path("results") / "feature_visualizations.json", result
            )
        return result


class SectionContext:
    def __init__(
        self, workspace: FeatureWorkspace, arm: Arm, section: FeatureSection, model: nn.Module
    ) -> None:
        self.workspace, self.arm, self.section, self.model = workspace, arm, section, model
        self.config, self.device, self.identity = (
            workspace.config,
            workspace.device,
            workspace.identity,
        )
        self.references, self.anchors, self.progress = (
            workspace.references,
            workspace.anchors,
            workspace.progress,
        )
        self.figures: list[dict[str, Any]] = []
        self.dependencies: list[dict[str, str]] = []

    def _dependency(self, arm: str, section: str) -> SectionData:
        result = self.workspace._load(arm, section)
        if result is None:
            raise FeatureEvidenceError(f"missing prerequisite: {arm}/{section}")
        path = self.workspace._path(arm, section)
        self.dependencies.append(
            {"path": str(path.relative_to(self.config.root)), "sha256": sha256_file(path)}
        )
        return result[1]

    def reference_pixels(self) -> torch.Tensor:
        return self.workspace._pixels(self.references)

    def reference_probe(self, *, initial: bool = False) -> dict[str, dict[str, torch.Tensor]]:
        from src.feature_visualization import LAYERS, _np, _probe

        arm = "initial" if initial else self.arm
        if self.workspace._load(arm, "probe") is None:
            if initial:
                raise FeatureEvidenceError("initial calibration baseline was not prepared")
            probes = _probe(
                self.model,
                self.reference_pixels(),
                self.config.integer("training", "evaluation_batch_size"),
            )
            self.workspace._save(
                arm,
                "probe",
                SectionData(
                    arrays={
                        f"{index}_{name}": _np(probes[layer][name])
                        for index, layer in enumerate(LAYERS)
                        for name in ("means", "peaks", "positions")
                    }
                ),
            )
        data = self._dependency(arm, "probe")
        return {
            layer: {
                name: torch.from_numpy(data.arrays[f"{index}_{name}"])
                for name in ("means", "peaks", "positions")
            }
            for index, layer in enumerate(LAYERS)
        }

    def selected_channels(self) -> dict[str, list[int]]:
        count = int(self.config.section("feature_visualization")["channels_per_layer"])
        return {
            layer: np.argsort(-probe["peaks"].numpy().mean(axis=0), kind="stable")[:count].tolist()
            for layer, probe in self.reference_probe().items()
        }

    def stage_channels(self) -> dict[str, list[int]]:
        from src.stage_walkthrough import rank_stage_channels

        if self.workspace._load(self.arm, "stage_channels") is None:
            selected = rank_stage_channels(
                self.model,
                self.reference_pixels(),
                self.config.integer("training", "evaluation_batch_size"),
            )
            self.workspace._save(
                self.arm, "stage_channels", SectionData(payload={"channels": selected})
            )
        return cast(
            dict[str, list[int]], self._dependency(self.arm, "stage_channels").payload["channels"]
        )

    def initial_kernels(self) -> torch.Tensor:
        return torch.from_numpy(self._dependency("initial", "kernels").arrays["kernels"])

    def anchor_pixels(self, index: int) -> torch.Tensor:
        from src.feature_visualization import _single_image_pixels

        pixels = PetRecordDataset([self.anchors[index]], self.config, training=False)[0][0]
        return _single_image_pixels(pixels, self.device)

    def dependency(self, section: FeatureSection) -> SectionData:
        return self._dependency(self.arm, section)

    def register(
        self,
        figure: Any,
        name: str,
        caption: str,
        kind: str,
        *,
        shareable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from src.feature_visualization import _save_figure_record

        path = (
            self.config.project_path("figures")
            / "features"
            / self.identity[:16]
            / self.section
            / f"{self.arm}-{name}.png"
        )
        info = {
            "identity": self.identity,
            "config_sha256": self.config.sha256,
            "checkpoint_sha256": self.workspace.checkpoints[self.arm],
            "sample_id": "calibration-selected",
        }
        info.update(metadata or {})
        record = _save_figure_record(
            figure,
            path,
            self.config.root,
            arm=self.arm,
            caption=caption,
            kind=kind,
            shareable=shareable,
            metadata=info,
        )
        record["section"] = self.section
        self.figures.append(record)


def compute_feature_sections(
    config: ExperimentConfig,
    device: torch.device,
    *,
    progress: bool = True,
    section: FeatureSection | None = None,
) -> dict[str, Any]:
    if section is not None and section not in FEATURE_SECTIONS:
        raise FeatureEvidenceError(f"unknown feature section: {section}")
    with _preserved_randomness(device):
        return _compute_feature_sections(config, device, progress=progress, section=section)


def _compute_feature_sections(
    config: ExperimentConfig,
    device: torch.device,
    *,
    progress: bool,
    section: FeatureSection | None,
) -> dict[str, Any]:
    workspace = FeatureWorkspace(config, device, progress)
    result: dict[str, Any] | None = None
    selected = (section,) if section is not None else FEATURE_SECTIONS
    for requested in selected:
        for arm in ARMS:
            try:
                workspace.run(arm, requested)
            except BaseException:
                # Preserve completed prerequisite receipts without substituting
                # an incomplete parent analysis or masking its original error.
                with suppress(Exception):
                    workspace.manifest()
                raise
            # Publish progress even when hydrated from cache; no unrelated computation.
            result = workspace.manifest()
    assert result is not None
    return result


def validate_feature_evidence(
    config: ExperimentConfig, result: dict[str, Any], *, require_complete: bool = True
) -> None:
    if result.get("config_sha256") != config.sha256:
        raise FeatureEvidenceError("feature evidence belongs to another configuration")
    if "identity" not in result and "section_receipts" not in result:
        # Older minimal report records remain readable; real identity-bearing
        # prior-method evidence is not silently relabelled as the staged run.
        if result.get("status", "complete") != "complete":
            raise FeatureEvidenceError("feature evidence is incomplete")
        return
    workspace = FeatureWorkspace(
        config,
        torch.device(result.get("device", "mps")),
        False,
        read_only=True,
    )
    if (
        result.get("identity") != workspace.identity
        or result.get("method_sha256") != workspace.method_sha256
    ):
        raise FeatureEvidenceError(
            "feature evidence has stale method, data, selection or checkpoint identity"
        )
    if require_complete and (
        result.get("status") != "complete" or result.get("completion", {}).get("missing")
    ):
        raise FeatureEvidenceError(
            "feature evidence is incomplete; finish all section 10 subsections"
        )
    receipts = result.get("section_receipts", {})
    registered_figures: list[dict[str, Any]] = []
    for arm in ARMS:
        registered = receipts.get(arm, {})
        if not isinstance(registered, dict) or not set(registered).issubset(FEATURE_SECTIONS):
            raise FeatureEvidenceError(f"invalid feature section registry for {arm}")
        if require_complete and set(registered) != set(FEATURE_SECTIONS):
            raise FeatureEvidenceError(f"feature evidence is missing sections for {arm}")
        for section in FEATURE_SECTIONS:
            if section not in registered:
                continue
            receipt = registered[section]
            _inside(config, receipt["path"], receipt["sha256"])
            cached = workspace._load(arm, section)
            if cached is None:
                raise FeatureEvidenceError(f"stale feature measurements for {arm}/{section}")
            registered_figures.extend(cached[0]["figures"])
        if require_complete and (
            result.get("arms", {}).get(arm, {}).get("checkpoint_sha256")
            != workspace.checkpoints[arm]
            or result.get("arms", {}).get(arm, {}).get("state_unchanged") is not True
        ):
            raise FeatureEvidenceError(
                f"feature model state or checkpoint identity is invalid for {arm}"
            )
    if result.get("figures", []) != registered_figures:
        raise FeatureEvidenceError("feature figure list does not match registered section receipts")
    for figure in result.get("figures", []):
        if figure.get("identity") != workspace.identity:
            raise FeatureEvidenceError("feature image has a different scientific identity")
        _inside(config, figure["path"], figure["sha256"])
