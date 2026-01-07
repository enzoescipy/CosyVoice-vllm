import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm


def cosyvoice2_example():
    """ CosyVoice2 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', '앞으로 더 멋진 사람이 되길 바랄게.', './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_example():
    """ CosyVoice3 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', 'You are a helpful assistant.<|endofprompt|>앞으로 더 멋진 사람이 되길 바랄게.',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
            continue


def main():
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()
