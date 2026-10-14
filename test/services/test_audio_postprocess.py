from app.services.audio_postprocess import build_filter


def test_audio_filter_has_loudness_and_clarity_filters():
    value = build_filter({"clarity": 50, "target_lufs": -16, "speed": 1.0, "pitch_semitones": 0})
    assert "highpass" in value
    assert "equalizer" in value
    assert "loudnorm=I=-16:TP=-1.0" in value


def test_speed_and_pitch_are_applied():
    value = build_filter({"clarity": 0, "target_lufs": -16, "speed": 1.1, "pitch_semitones": 1})
    assert "asetrate" in value
    assert "atempo=1.1000" in value


def test_processing_never_allows_same_source_and_destination(tmp_path):
    from app.services.audio_postprocess import process_audio
    source = tmp_path / "same.wav"
    source.write_bytes(b"RIFF" + b" " * 40)
    try:
        process_audio(source, source)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("Expected same-source protection")
