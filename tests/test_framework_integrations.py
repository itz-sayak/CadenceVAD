from __future__ import annotations

import asyncio
import platform
from pathlib import Path

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.vad import VADEventType
from pipecat.audio.vad.vad_analyzer import VADState

from cadencevad.integrations.livekit import CadenceVadLiveKit
from cadencevad.integrations.pipecat import CadenceVadPipecatAnalyzer
from cadencevad.runtime import OnnxStreamingVadModel


def _model_path() -> Path:
    return Path("models/cadencevad-v0.1/cadencevad-stream.onnx")


def _owner() -> OnnxStreamingVadModel:
    path = _model_path()
    if not path.exists():
        pytest.skip("retained ONNX model is not present")
    return OnnxStreamingVadModel(path)


def test_pipecat_adapter_processes_16khz_and_8khz_pcm() -> None:
    owner = _owner()
    analyzer = CadenceVadPipecatAnalyzer(owner=owner)

    analyzer.set_sample_rate(16_000)
    probability_16k = analyzer.voice_confidence(
        np.zeros(160, dtype=np.int16).tobytes()
    )
    assert 0.0 <= probability_16k <= 1.0

    analyzer.set_sample_rate(8_000)
    probability_8k = analyzer.voice_confidence(
        np.zeros(80, dtype=np.int16).tobytes()
    )
    assert 0.0 <= probability_8k <= 1.0


def test_pipecat_adapter_loads_the_bundled_model_by_default() -> None:
    analyzer = CadenceVadPipecatAnalyzer()
    analyzer.set_sample_rate(16_000)

    probability = analyzer.voice_confidence(
        np.zeros(160, dtype=np.int16).tobytes()
    )

    assert 0.0 <= probability <= 1.0


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires Apple Accelerate")
def test_pipecat_adapter_loads_the_bundled_native_runtime() -> None:
    analyzer = CadenceVadPipecatAnalyzer.load_native()
    analyzer.set_sample_rate(16_000)

    probability = analyzer.voice_confidence(
        np.zeros(160, dtype=np.int16).tobytes()
    )

    assert 0.0 <= probability <= 1.0


def test_pipecat_adapter_rejects_unsupported_audio_rate() -> None:
    analyzer = CadenceVadPipecatAnalyzer(owner=_owner())

    with pytest.raises(ValueError, match="8 kHz or 16 kHz"):
        analyzer.set_sample_rate(48_000)

    assert analyzer.sample_rate == 0


def test_pipecat_adapter_reset_clears_framework_and_model_state() -> None:
    analyzer = CadenceVadPipecatAnalyzer(owner=_owner())
    analyzer.set_sample_rate(16_000)
    analyzer._vad_state = VADState.SPEAKING
    analyzer._vad_starting_count = 2
    analyzer._vad_stopping_count = 3
    analyzer._prev_volume = 0.75
    analyzer._vad_buffer = b"buffered"

    analyzer.reset()

    assert analyzer._vad_state is VADState.QUIET
    assert analyzer._vad_starting_count == 0
    assert analyzer._vad_stopping_count == 0
    assert analyzer._prev_volume == 0
    assert analyzer._vad_buffer == b""


def test_pipecat_adapter_cleanup_releases_its_executor() -> None:
    analyzer = CadenceVadPipecatAnalyzer(owner=_owner())

    asyncio.run(analyzer.cleanup())
    asyncio.run(analyzer.cleanup())

    assert analyzer._executor._shutdown


def test_pipecat_adapter_runs_current_async_analyzer_contract() -> None:
    async def scenario() -> VADState:
        analyzer = CadenceVadPipecatAnalyzer(owner=_owner())
        analyzer.set_sample_rate(16_000)
        state = await analyzer.analyze_audio(
            np.zeros(160, dtype=np.int16).tobytes()
        )
        await analyzer.cleanup()
        return state

    assert asyncio.run(scenario()) is VADState.QUIET


def test_livekit_adapter_emits_inference_for_16khz_audio() -> None:
    async def scenario() -> None:
        stream = CadenceVadLiveKit(_owner()).stream()
        stream.push_frame(
            rtc.AudioFrame(
                data=np.zeros(160, dtype=np.int16).tobytes(),
                sample_rate=16_000,
                num_channels=1,
                samples_per_channel=160,
            )
        )
        event = await asyncio.wait_for(anext(stream), timeout=2)
        assert event.type == VADEventType.INFERENCE_DONE
        assert event.samples_index == 160
        assert 0.0 <= event.probability <= 1.0
        stream.end_input()
        async for _ in stream:
            pass

    asyncio.run(scenario())


def test_livekit_adapter_reports_onnx_provider() -> None:
    assert CadenceVadLiveKit(_owner()).provider == "CadenceVAD ONNX"


def test_livekit_adapter_loads_the_bundled_model_by_default() -> None:
    vad = CadenceVadLiveKit.load()

    assert vad.provider == "CadenceVAD ONNX"
    assert vad.model == "cadencevad-v0.1"


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires Apple Accelerate")
def test_livekit_adapter_loads_the_bundled_native_runtime() -> None:
    vad = CadenceVadLiveKit.load_native()

    assert vad.provider == "CadenceVAD Accelerate"


def test_livekit_adapter_resamples_48khz_audio_to_one_model_hop() -> None:
    async def scenario() -> None:
        stream = CadenceVadLiveKit(_owner()).stream()
        stream.push_frame(
            rtc.AudioFrame(
                data=np.zeros(480, dtype=np.int16).tobytes(),
                sample_rate=48_000,
                num_channels=1,
                samples_per_channel=480,
            )
        )
        event = await asyncio.wait_for(anext(stream), timeout=2)
        assert event.type == VADEventType.INFERENCE_DONE
        assert event.samples_index == 160
        stream.end_input()
        async for _ in stream:
            pass

    asyncio.run(scenario())
