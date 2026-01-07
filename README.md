# CosyVoice-vllm

A minimal fork of [Lourdle/CosyVoice](https://github.com/Lourdle/CosyVoice) to make `load_vllm=True` actually work.

## Why This Fork Exists

The original [Lourdle/CosyVoice](https://github.com/Lourdle/CosyVoice) provides excellent ONNX support, but attempting to use the `load_vllm=True` option often results in argument-related errors due to version mismatches or unsupported parameters in the underlying engine.

This fork is a "surgical fix" to address these specific compatibility issues, ensuring that users can leverage both ONNX and vLLM acceleration in a unified environment.

## What's Different

- **vLLM Compatibility Patch:** Fixed internal argument errors when initializing the model with `load_vllm=True`.
- **Flexible Dependencies:** Removed strict version pins in `requirements.txt` to allow installation in modern Python/WSL environments without dependency hell.
- **Korean Localization:** Translated example scripts (`example.py`, `vllm_example.py`) and the WebUI (`webui.py`) into Korean for easier local testing.
- **Pruned Requirements:** Streamlined dependencies to focus on inference tasks.

## What This Fork Does NOT Do

This is a developer-centric fork focused on the core library.
- **No Full-Stack Ambition:** We do not focus on Docker deployment, Triton serving, or complex server-side migrations.
- **Library Focus:** The goal is to make `import cosyvoice` work reliably on your local machine so you can build your own tools on top of it.

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/enzoescipy/CosyVoice-vllm.git
cd CosyVoice-vllm
conda create -n cosyvoice-vllm python=3.10
conda activate cosyvoice-vllm
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run
Check `example.py` for general usage or `vllm_example.py` for vLLM-specific testing.
```bash
python vllm_example.py
```

## Acknowledgements

This project is built upon the great work of:
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - The original revolutionary TTS model.
- [Lourdle/CosyVoice](https://github.com/Lourdle/CosyVoice) - Providing the essential ONNX implementation.

## License
Same as original projects (Apache 2.0). See LICENSE for details.
