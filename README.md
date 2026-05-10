# OmniVoice 4090 Test

Minimal files for testing OmniVoice zero-shot cloning on an RTX 4090.

## Setup

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run jupyter notebook
```

Open `OmniVoice_Zero_Shot_Cloning.ipynb` and run the cells.

To run the timing benchmark:

```powershell
uv run python benchmark_omnivoice_zero_shot.py --num-step 8 --speed 1.18 --warmup
uv run python benchmark_omnivoice_zero_shot.py --num-step 2 --speed 1.18 --warmup
uv run python benchmark_omnivoice_zero_shot.py --num-step 1 --speed 1.18 --warmup
```

`sin_2282_8643512444.wav` is the reference clip with matching `ref_text` already set in the notebook.
`homelands_omnivoice_sample.wav` is included as an optional reference clip, but its exact transcript must be set before using it.
`expected_target.wav` is the original extracted audio for the target sentence, included only for comparison.
