# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['사전학습 음색', '3초 즉석 복제', '다국어 복제', '자연어 제어']
instruct_dict = {'사전학습 음색': '1. 사전학습 음색을 선택하세요\n2. 생성 오디오 버튼을 클릭하세요',
                 '3초 즉석 복제': '1. prompt 오디오 파일을 선택하거나, prompt 오디오를 녹음하세요. 30초 이내로 유지하세요. 둘 다 제공 시 파일 우선.\n2. prompt 텍스트를 입력하세요\n3. 생성 오디오 버튼을 클릭하세요',
                 '다국어 복제': '1. prompt 오디오 파일을 선택하거나, prompt 오디오를 녹음하세요. 30초 이내로 유지하세요. 둘 다 제공 시 파일 우선.\n2. 생성 오디오 버튼을 클릭하세요',
                 '자연어 제어': '1. 사전학습 음색을 선택하세요\n2. instruct 텍스트를 입력하세요\n3. 생성 오디오 버튼을 클릭하세요'}
stream_mode_list = [('아니오', False), ('예', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['자연어 제어']:
        if instruct_text == '':
            gr.Warning('자연어 제어 모드를 사용 중입니다, instruct 텍스트를 입력하세요')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('자연어 제어 모드를 사용 중입니다, prompt 오디오/prompt 텍스트는 무시됩니다')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['다국어 복제']:
        if instruct_text != '':
            gr.Info('다국어 복제 모드를 사용 중입니다, instruct 텍스트는 무시됩니다')
        if prompt_wav is None:
            gr.Warning('다국어 복제 모드를 사용 중입니다, prompt 오디오를 제공하세요')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('다국어 복제 모드를 사용 중입니다, 합성 텍스트와 prompt 텍스트가 다른 언어인지 확인하세요')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3초 즉석 복제', '다국어 복제']:
        if prompt_wav is None:
            gr.Warning('prompt 오디오가 비어 있습니다, prompt 오디오 입력을 잊으셨나요?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt 오디오 샘플레이트 {}가 {}보다 낮습니다'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['사전학습 음색']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('사전학습 음색 모드를 사용 중입니다, prompt 텍스트/prompt 오디오/instruct 텍스트는 무시됩니다!')
        if sft_dropdown == '':
            gr.Warning('사용 가능한 사전학습 음색이 없습니다!')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3초 즉석 복제']:
        if prompt_text == '':
            gr.Warning('prompt 텍스트가 비어 있습니다, prompt 텍스트 입력을 잊으셨나요?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('3초 즉석 복제 모드를 사용 중입니다, 사전학습 음색/instruct 텍스트는 무시됩니다!')

    if mode_checkbox_group == '사전학습 음색':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3초 즉석 복제':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '다국어 복제':
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 코드 라이브러리 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    사전학습 모델 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 합성할 텍스트를 입력하고, 추론 모드를 선택한 후 지시 단계에 따라 작업하세요")

        tts_text = gr.Textbox(label="합성할 텍스트 입력", lines=1, value="안녕하세요, 저는 Tongyi 실험실 음성 팀에서 새로 출시한 생성형 음성 대형 모델입니다. 편안하고 자연스러운 음성 합성 기능을 제공합니다.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='추론 모드 선택', value=inference_mode_list[0])
            instruction_text = gr.Text(label="작업 단계", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='사전학습 음색 선택', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='스트리밍 추론 여부', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="속도 조절(비스트리밍 추론만 지원)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="랜덤 추론 시드")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='prompt 오디오 파일 선택, 샘플레이트 16kHz 이상 주의')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='prompt 오디오 녹음')
        prompt_text = gr.Textbox(label="prompt 텍스트 입력", lines=1, placeholder="prompt 텍스트 입력, prompt 오디오 내용과 일치해야 함. 자동 인식 미지원...", value='')
        instruct_text = gr.Textbox(label="instruct 텍스트 입력", lines=1, placeholder="instruct 텍스트 입력.", value='')

        generate_button = gr.Button("오디오 생성")

        audio_output = gr.Audio(label="합성 오디오", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--use_onnx',
                        action='store_true',
                        default=False,
                        help='use onnx model. Only support CosyVoice2-0.5B model. If set, will not support streaming inference.')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir, load_onnx=args.use_onnx)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()