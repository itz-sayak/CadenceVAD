from __future__ import annotations

import json
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from .benchmark import load_checkpoint
from .model import CadenceVad


def _strip_graph_debug_metadata(graph: onnx.GraphProto) -> None:
    graph.name = "cadencevad"
    graph.doc_string = ""
    for value in (*graph.input, *graph.output, *graph.value_info):
        value.doc_string = ""
    for node in graph.node:
        node.doc_string = ""
        if hasattr(node, "metadata_props"):
            node.ClearField("metadata_props")
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _strip_graph_debug_metadata(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    _strip_graph_debug_metadata(child)


def _sanitize_onnx_metadata(path: Path) -> None:
    """Remove exporter traces, namespaces, and local paths from a public graph."""
    model = onnx.load(path)
    model.doc_string = ""
    model.ClearField("metadata_props")
    _strip_graph_debug_metadata(model.graph)
    onnx.save_model(model, path, save_as_external_data=False)
    exporter_sidecar = path.with_name(f"{path.name}.data")
    if exporter_sidecar.is_file():
        exporter_sidecar.unlink()


class OfflineExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor) -> Tensor:
        return self.model(features)["speech_logits"]


class StreamingExportWrapper(nn.Module):
    def __init__(self, model: CadenceVad) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        feature: Tensor,
        recurrent: Tensor,
        *state: Tensor,
    ) -> tuple[Tensor, ...]:
        # When the model carries a running-norm cache it is threaded first, ahead
        # of the convolution caches, matching the input/output name order below.
        running: Tensor | None = None
        if self.model.running_norm is not None:
            running, *rest = state
            convolution = tuple(rest)
            feature, running = self.model.running_norm.stream(feature, running)
        else:
            convolution = state
        encoded = self.model._input(feature)
        next_caches: list[Tensor] = []
        for block, cache in zip(self.model.blocks, convolution, strict=True):
            encoded, next_cache = block.stream(encoded, cache)
            next_caches.append(next_cache)
        encoded, next_recurrent = self.model.recurrent(encoded, recurrent)
        speech_logits = self.model._heads(encoded)["speech_logits"]
        if running is not None:
            return (speech_logits, next_recurrent, running, *next_caches)
        return (speech_logits, next_recurrent, *next_caches)


def _set_symbolic_output_shapes(
    path: Path, output_shapes: dict[str, tuple[str | int, ...]]
) -> None:
    model = onnx.load(path)
    for output in model.graph.output:
        requested = output_shapes.get(output.name)
        if requested is None:
            continue
        dimensions = output.type.tensor_type.shape.dim
        if len(dimensions) != len(requested):
            raise ValueError(f"cannot assign shape {requested} to output {output.name}")
        for dimension, value in zip(dimensions, requested, strict=True):
            if isinstance(value, str):
                dimension.dim_param = value
            else:
                dimension.dim_value = value
    onnx.save(model, path)


def _write_metadata(path: Path, config: dict[str, object], mode: str) -> None:
    metadata = {
        "feature": config["feature"],
        "model": config["model"],
        "detector": config["detector"],
        "mode": mode,
        "note": "Graph accepts precomputed causal features.",
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def export_offline_onnx(
    checkpoint_path: str | Path,
    destination: str | Path,
    sequence_frames: int = 100,
) -> Path:
    config, model = load_checkpoint(checkpoint_path)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = OfflineExportWrapper(model).eval()
    # A dimension whose example size is 1 is specialized by torch.export.
    # Use batch 2 so the exported batch axis remains genuinely dynamic.
    example = torch.zeros(2, sequence_frames, config.model.feature_dim)
    batch = torch.export.Dim("batch")
    frames = torch.export.Dim("frames")
    torch.onnx.export(
        wrapper,
        (example,),
        output,
        input_names=["features"],
        output_names=["speech_logits"],
        dynamic_shapes={"features": {0: batch, 1: frames}},
        opset_version=18,
        dynamo=True,
    )
    _set_symbolic_output_shapes(output, {"speech_logits": ("batch", "frames")})
    _sanitize_onnx_metadata(output)
    _write_metadata(output, config.to_dict(), "offline")
    return output


def export_streaming_onnx(
    checkpoint_path: str | Path,
    destination: str | Path,
) -> Path:
    config, model = load_checkpoint(checkpoint_path)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StreamingExportWrapper(model).eval()
    state = model.initial_state(2, "cpu")
    feature = torch.zeros(2, 1, config.model.feature_dim)
    running = (state.input_norm,) if state.input_norm is not None else ()
    running_names = ["running_norm"] if running else []
    arguments = (feature, state.recurrent, *running, *state.convolution)
    input_names = [
        "feature",
        "recurrent",
        *running_names,
        *[f"cache_{index}" for index in range(len(state.convolution))],
    ]
    output_names = [
        "speech_logits",
        "next_recurrent",
        *[f"next_{name}" for name in running_names],
        *[f"next_cache_{index}" for index in range(len(state.convolution))],
    ]
    batch = torch.export.Dim("batch")
    # ``forward(feature, recurrent, *state)`` collapses every trailing tensor into
    # one varargs node, so the running-norm cache belongs inside that tuple rather
    # than beside it.
    dynamic_shapes = (
        {0: batch},
        {1: batch},
        tuple({0: batch} for _ in (*running, *state.convolution)),
    )
    torch.onnx.export(
        wrapper,
        arguments,
        output,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        dynamo=True,
    )
    symbolic_shapes: dict[str, tuple[str | int, ...]] = {
        "speech_logits": ("batch", 1),
        "next_recurrent": (1, "batch", config.model.recurrent_dim),
    }
    if model.running_norm is not None:
        symbolic_shapes["next_running_norm"] = (
            "batch",
            model.running_norm.output_dim,
        )
    for index, block in enumerate(model.blocks):
        symbolic_shapes[f"next_cache_{index}"] = (
            "batch",
            config.model.hidden_dim,
            block.cache_size,
        )
    _set_symbolic_output_shapes(output, symbolic_shapes)
    _sanitize_onnx_metadata(output)
    _write_metadata(output, config.to_dict(), "streaming")
    return output


def export_onnx(
    checkpoint_path: str | Path,
    destination: str | Path,
    sequence_frames: int = 100,
    mode: str = "streaming",
) -> Path:
    if mode == "streaming":
        return export_streaming_onnx(checkpoint_path, destination)
    if mode == "offline":
        return export_offline_onnx(checkpoint_path, destination, sequence_frames)
    raise ValueError(f"unsupported export mode: {mode}")
