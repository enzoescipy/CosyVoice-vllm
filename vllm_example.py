
# import sys
# sys.path.append('third_party/Matcha-TTS')
# from vllm import ModelRegistry
# from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
# ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# from cosyvoice.cli.cosyvoice import AutoModel
# from cosyvoice.utils.common import set_all_random_seed
# from tqdm import tqdm


# def cosyvoice2_example():
#     """ CosyVoice2 vllm usage
#     """
#     cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
#     for i in tqdm(range(100)):
#         set_all_random_seed(i)
#         for _, _ in enumerate(cosyvoice.inference_zero_shot('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', '앞으로 더 멋진 사람이 되길 바랄게.', './asset/zero_shot_prompt.wav', stream=False)):
#             continue


# def cosyvoice3_example():
#     """ CosyVoice3 vllm usage
#     """
#     cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
#     for i in tqdm(range(100)):
#         set_all_random_seed(i)
#         for _, _ in enumerate(cosyvoice.inference_zero_shot('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', 'You are a helpful assistant.<|endofprompt|>앞으로 더 멋진 사람이 되길 바랄게.',
#                                                             './asset/zero_shot_prompt.wav', stream=False)):
#             continue


# def main():
    # cosyvoice2_example()
    # cosyvoice3_example()


import os
import torch

## setup environ for sure. this fixes the gpu usage into cuda:0.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._device = None

    def load(self):
        # 1. Download model weights from ModelScope
        # This will download the model to a local cache directory
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            self._model = AutoModel(
                model_dir='pretrained_models/Fun-CosyVoice3-0.5B', 
                load_trt=True, load_vllm=True, fp16=True)
            print(f"Model successfully loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # 2. Set device
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self._device}")

    def predict(self, model_input):
        import base64
        import io
        import numpy as np
        from scipy.io import wavfile

        # 0. Extract input text and optional instruction
        text = model_input.get('text', '안녕하세요, 위더. 새로운 목소리 시스템이 준비되었습니다.')
        instruct = model_input.get('instruct', 'You are a helpful assistant.')

        try:
            # 1. CosyVoice3 Instruct logic:
            # Pattern: <Prompt Text> + <Prompt Audio>
            # Based on example: inference_instruct2(text, prompt_text, prompt_wav)
            # prompt_text should end with <|endofprompt|>
            prompt_text = f"{instruct}<|endofprompt|>" + text
            prompt_wav = './asset/zero_shot_prompt.wav'

            # 2. Perform inference
            # AutoModel.inference_instruct2 returns a generator
            outputs_gen = self._model.inference_cross_lingual(
                prompt_text,
                prompt_wav,
                stream=False
            )

            # 3. Extract and combine waveforms (in case generator yields multiple chunks)
            waveforms = []
            for output in outputs_gen:
                if 'tts_speech' in output:
                    waveforms.append(output['tts_speech'].cpu().numpy())

            if not waveforms:
                return {'error': 'No speech generated'}

            waveform = np.concatenate(waveforms, axis=1).flatten()

            # 4. Standardize audio format (CosyVoice3 typically uses 22050Hz)
            sample_rate = self._model.sample_rate if hasattr(
                self._model, 'sample_rate') else 22050

            # 5. Convert to WAV in memory
            # Note: wave data should be float32 or int16.
            # Scipy wavfile handles float32 (-1.0 to 1.0) or int16.
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, waveform)

            # 6. Encode to base64
            buffer.seek(0)
            encoded_audio = base64.b64encode(buffer.read()).decode('utf-8')

            return {'audio': encoded_audio}

        except Exception as e:
            print(f"Error during prediction: {e}")
            return {'error': str(e)}




if __name__ == '__main__':

    ## initial setup
    from huggingface_hub import snapshot_download
    if not os.path.isdir("pretrained_models/Fun-CosyVoice3-0.5B"):
        snapshot_download('Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
        snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B', allow_patterns='*')
    if not os.path.isdir("pretrained_models/CosyVoice-ttsfrd"):
        snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd', allow_patterns='*')
    ## operation
    import time
    import base64

    model = Model()
    model.load()

    test_input = {
        'text': '오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.',
        'instruct': 'You are a helpful assistant.'
    }

    start_time = time.time()
    result = model.predict(test_input)
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")

    if 'audio' in result:
        audio_bytes = base64.b64decode(result['audio'])
        sample_rate = 22050  # Default for CosyVoice3; can be overridden if model exposes it
        output_filename = 'test_output.wav'

        # Write to WAV file
        # Note: Assuming audio_bytes is the raw WAV data from the BytesIO in predict
        with open(output_filename, 'wb') as f:
            f.write(audio_bytes)
        print(f"[Success] Audio saved to {output_filename}")
    else:
        print(f"[Error] {result.get('error', 'Unknown error')}")