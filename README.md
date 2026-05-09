# OmniVoice 4090 Test

Minimal files for testing OmniVoice zero-shot cloning on an RTX 4090.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install omnivoice soundfile notebook ipywidgets
jupyter notebook
```

Open `OmniVoice_Zero_Shot_Cloning.ipynb` and run the cells.

To run the timing benchmark:

```powershell
python benchmark_omnivoice_zero_shot.py --num-step 8 --speed 1.18 --warmup
python benchmark_omnivoice_zero_shot.py --num-step 2 --speed 1.18 --warmup
python benchmark_omnivoice_zero_shot.py --num-step 1 --speed 1.18 --warmup
```

`sin_2282_8643512444.wav` is the reference clip with matching `ref_text` already set in the notebook.
`homelands_omnivoice_sample.wav` is included as an optional reference clip, but its exact transcript must be set before using it.
`expected_target.wav` is the original extracted audio for the target sentence, included only for comparison.
