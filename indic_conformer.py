"""Transcribe a Hindi audio file using AI4Bharat IndicConformer."""

import argparse
from pathlib import Path


MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000


def transcribe(audio_path: Path) -> str:
    import soundfile as sf
    import torch
    from torchaudio.functional import resample
    from transformers import AutoModel

    samples, source_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("The audio file is empty.")

    waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
    if source_rate != SAMPLE_RATE:
        waveform = resample(waveform, source_rate, SAMPLE_RATE)

    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    with torch.no_grad():
        result = model(waveform, "hi", "ctc")

    if isinstance(result, (list, tuple)):
        result = result[0]
    return str(result).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path, help="Path to an audio file, such as a WAV")
    args = parser.parse_args()
    if not args.audio_file.is_file():
        parser.error(f"Audio file not found: {args.audio_file}")

    try:
        transcript = transcribe(args.audio_file)
    except Exception as exc:
        parser.exit(1, f"Transcription failed: {exc}\n")
    print(transcript)


if __name__ == "__main__":
    main()
