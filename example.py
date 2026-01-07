# Note : enzoescipy/CosyVoice-vllm encourage to see the vllm_example.py too.


import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
import sys
sys.path.append('third_party/Matcha-TTS')


def cosyvoice_example():
    """ CosyVoice 사용법, 자세한 내용은 https://fun-audio-llm.github.io/ 확인
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')
    # sft 사용법
    print(cosyvoice.list_available_spks())
    # 청크 스트리밍 추론을 위해 stream=True로 변경
    for i, j in enumerate(cosyvoice.inference_sft('안녕하세요, 저는 Tongyi 생성형 음성 언어모델입니다. 도와드릴까요?', '한국어 여성', stream=False)):
        torchaudio.save('sft_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')
    # zero_shot 사용법
    for i, j in enumerate(cosyvoice.inference_zero_shot('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', '앞으로 더 멋진 사람이 되길 바랄게.', './asset/zero_shot_prompt.wav')):
        torchaudio.save('zero_shot_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)
    # cross_lingual 사용법, 중국어/영어/일본어/광둥어/한국어용 <|zh|><|en|><|ja|><|yue|><|ko|>
    for i, j in enumerate(cosyvoice.inference_cross_lingual('<|ko|>그리고 나중에 그 회사를 완전히 인수하게 되면. 그래서 경영진을 일하게 하고, 가족에 들어오는 자산과 일치하는 이익을 유지하는 것이, 때때로 전체를 사지 않는 이유입니다.',
                                                            './asset/cross_lingual_prompt.wav')):
        torchaudio.save('cross_lingual_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)
    # vc 사용법
    for i, j in enumerate(cosyvoice.inference_vc('./asset/cross_lingual_prompt.wav', './asset/zero_shot_prompt.wav')):
        torchaudio.save('vc_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(
        model_dir='pretrained_models/CosyVoice-300M-Instruct')
    # instruct 사용법, <laughter></laughter><strong></strong>[laughter][breath] 지원
    for i, j in enumerate(cosyvoice.inference_instruct('도전을 마주할 때, 그는 비범한<strong>용기</strong>와 <strong>지혜</strong>를 보여줍니다.', '한국어 남성',
                                                       '테오 크림슨은 불타오르는 열정의 반항군 리더입니다. 정의를 위해 열정적으로 싸우지만, 충동적입니다.<|endofprompt|>')):
        torchaudio.save('instruct_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)


def cosyvoice2_example():
    """ CosyVoice2 사용법, 자세한 내용은 https://funaudiollm.github.io/cosyvoice2/ 확인
    """
    cosyvoice = AutoModel(
        model_dir='pretrained_models/CosyVoice2-0.5B', load_onnx=True)

    # 주의: https://funaudiollm.github.io/cosyvoice2 결과를 재현하려면 추론 시 text_frontend=False 추가
    # zero_shot 사용법
    # 주의: ONNX 모드에서는 스트리밍 추론(stream=True) 미지원, load_onnx=True 시 항상 stream=False 사용
    for i, j in enumerate(cosyvoice.inference_zero_shot('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', '앞으로 더 멋진 사람이 되길 바랄게.', './asset/zero_shot_prompt.wav')):
        torchaudio.save('zero_shot_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    # future use를 위한 zero_shot spk 저장
    assert cosyvoice.add_zero_shot_spk(
        '앞으로 더 멋진 사람이 되길 바랄게.', './asset/zero_shot_prompt.wav', 'my_zero_shot_spk') is True
    for i, j in enumerate(cosyvoice.inference_zero_shot('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', '', '', zero_shot_spk_id='my_zero_shot_spk')):
        torchaudio.save('zero_shot_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)
    cosyvoice.save_spkinfo()

    # 세밀한 제어, 지원 제어는 cosyvoice/tokenizer/tokenizer.py#L248 확인
    for i, j in enumerate(cosyvoice.inference_cross_lingual('그가 그 터무니없는 이야기를 하는 동안, 갑자기[laughter] 멈추었어요. 왜냐하면 자신도 웃기게 되었기 때문이에요[laughter].', './asset/zero_shot_prompt.wav')):
        torchaudio.save('fine_grained_control_{}.wav'.format(i),
                        j['tts_speech'], cosyvoice.sample_rate)

    # instruct 사용법
    for i, j in enumerate(cosyvoice.inference_instruct2('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', '제주도 사투리로 말해주세요.<|endofprompt|>', './asset/zero_shot_prompt.wav')):
        torchaudio.save('instruct_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    # bistream 사용법, generator를 입력으로 사용 가능, 텍스트 LLM 모델 입력 시 유용
    # 주의: LLM이 임의 문장 길이를 처리할 수 없으므로 기본 문장 분할 로직 필요
    def text_generator():
        yield '오랜만에 친구에게서 '
        yield '생일 선물을 받았어요.'
        yield '예상치 못한 깜짝 선물에 '
        yield '마음이 따뜻해지고, 절로 미소가 지어졌습니다.'
    for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '앞으로 더 멋진 사람이 되길 바랄게.', './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_bistream_{}.wav'.format(i),
                        j['tts_speech'], cosyvoice.sample_rate)


def cosyvoice3_example():
    """ CosyVoice3 사용법, 자세한 내용은 https://funaudiollm.github.io/cosyvoice3/ 확인
    """
    cosyvoice = AutoModel(
        model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_onnx=True)
    # zero_shot 사용법
    # 주의: ONNX 모드에서는 스트리밍 추론(stream=True) 미지원, load_onnx=True 시 항상 stream=False 사용
    for i, j in enumerate(cosyvoice.inference_zero_shot('간장 공장 공장장은 장 공장장이다.', 'You are a helpful assistant.<|endofprompt|>앞으로 더 멋진 사람이 되길 바랄게.',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    # 세밀한 제어, 지원 제어는 cosyvoice/tokenizer/tokenizer.py#L280 확인
    for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>[breath]그 세대 사람들은[breath] 시골에서 살다 보니 익숙해졌어요,[breath] 이웃들도 활발하고,[breath]음, 다들 친숙해요.[breath]',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('fine_grained_control_{}.wav'.format(i),
                        j['tts_speech'], cosyvoice.sample_rate)

    # instruct 사용법, 지원 제어는 cosyvoice/utils/common.py#L28 확인
    for i, j in enumerate(cosyvoice.inference_instruct2('좀 적어요, 보통 광복절이나 단오처럼 될 거예요.', 'You are a helpful assistant. 제주 사투리로 표현해주세요.<|endofprompt|>',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('instruct_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)
    for i, j in enumerate(cosyvoice.inference_instruct2('오랜만에 친구에게서 생일 선물을 받았어요. 예상치 못한 깜짝 선물에 마음이 따뜻해지고, 절로 미소가 지어졌습니다.', 'You are a helpful assistant. 가능한 한 빠른 말투로 한 문장을 말해주세요.<|endofprompt|>',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('instruct_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    # hotfix 사용법
    for i, j in enumerate(cosyvoice.inference_zero_shot('시민들도 인스타그램, 페이스북, X(트위터) 등 SNS를 통해 보도에 [j][ǐ] 좋은 평가를 내렸습니다.', 'You are a helpful assistant.<|endofprompt|>앞으로 더 멋진 사람이 되길 바랄게.',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('hotfix_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)


def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()
