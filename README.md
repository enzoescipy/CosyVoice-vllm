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

- **development enviroment setting**
 > apt-get update
 > apt-get install -y git espeak-ng ffmpeg
 please note that **cuDNN 9.* and CUDA 12.* is needed** for onnx.
 if you are using the hosted ML product instance, cuda-toolkit-12 might be ok not to installed.

- **Install pytorch FIRST !!!**
 torch serize installation : https://pytorch.org/get-started/previous-versions
 grab the one of <=v2.9.0 versions, which is : 
 > pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
 CUDA version is depends on the local running instance's gpu CUDA version.

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run
Check `example.py` for general usage or `vllm_example.py` for vLLM-specific testing.
Strongly recommended to see `vllm_example.py`'s following setup section : 

```python
if __name__ == '__main__':
    from huggingface_hub import snapshot_download
    if not os.path.isdir("pretrained_models/Fun-CosyVoice3-0.5B"):
        snapshot_download('Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
        snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B', allow_patterns='*')
    if not os.path.isdir("pretrained_models/CosyVoice-ttsfrd"):
        snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd', allow_patterns='*')
        
        # ttsfrd resource extraction
        import zipfile
        ttsfrd_dir = 'pretrained_models/CosyVoice-ttsfrd'
        zip_path = os.path.join(ttsfrd_dir, 'resource.zip')
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {ttsfrd_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ttsfrd_dir)
            print("Extraction completed.")
        else:
            print(f"Warning: {zip_path} not found for extraction.")

        # ttsfrd installation (can be simplified with just the:
        # > pip install pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl)
        whl_path = os.path.join('pretrained_models','CosyVoice-ttsfrd','ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl')
        subprocess.check_call([sys.executable, "-m", "pip", "install", whl_path])
```

or just run the

```bash
python vllm_example.py
```

## Acknowledgements

This project is built upon the great work of:
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - The original revolutionary TTS model.
- [Lourdle/CosyVoice](https://github.com/Lourdle/CosyVoice) - Providing the essential ONNX implementation.

## License
Same as original projects (Apache 2.0). See LICENSE for details.
