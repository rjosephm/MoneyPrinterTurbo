# Kaggle MoneyPrinterTurbo + IndicF5

## Kaggle setup
Add these under **Add-ons → Secrets**:
- `GITHUB_TOKEN`: fine-grained token with read-only Contents access to your private IndicF5 repository.
- `HF_TOKEN`: Hugging Face token after accepting the access conditions for `ai4bharat/IndicF5`.

Configure the repository slug in `Kaggle_Master.ipynb` after you create the private repository and its `kaggle` branch. Tokens must never be written into notebook source or printed.

## Reference voice
IndicF5 inference needs a reference WAV and its exact transcript. Set `INDICF5_REFERENCE_AUDIO` to a WAV path accessible in the Kaggle session and `INDICF5_REFERENCE_TEXT` to the exact spoken words. Model weights do not replace this requirement.

## Processes
- `indicf5_api.py`: FastAPI service on port 8765.
- `mpt_kaggle.py`: separate Streamlit launcher on port 8501.
- `generate_from_json.py`: CLI generation using the Manual Timeline render functions.

The master notebook runs smoke checks before rendering. Kaggle sessions are temporary; attach a private Kaggle Dataset with cached model files if you want to avoid repeated model downloads. A public tunnel can expose the Streamlit UI to anyone who has the URL; keep it disabled unless needed.
