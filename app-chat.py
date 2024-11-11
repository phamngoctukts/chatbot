import speech_recognition as sr
from gtts import gTTS
import gradio as gr
from io import BytesIO
import numpy as np
from dataclasses import dataclass, field
import time
import traceback
from pydub import AudioSegment
import librosa
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
import torch
from huggingface_hub import login
import os
from PIL import Image
from threading import Thread
tk = os.environ.get("HF_TOKEN")
#login("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")
ckpt = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(ckpt,torch_dtype=torch.bfloat16).to("cuda")
processor = AutoProcessor.from_pretrained(ckpt)
r = sr.Recognizer()

@dataclass
class AppState:
    stream: np.ndarray | None = None
    image: dict = field(default_factory=dict)
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool =  False
    stopped: bool = False
    message: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    conversation: list = field(default_factory=list)
    textout: str = ""

def run_vad(ori_audio, sr):
    _st = time.time()
    try:
        audio = ori_audio
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = 16000
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate
        if sr != sampling_rate:
            # resample to original sampling rate
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        vad_audio_bytes = vad_audio.tobytes()
        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)

def determine_pause(audio:np.ndarray,sampling_rate:int,state:AppState) -> bool:
    """Phát hiện tạm dừng trong âm thanh."""
    temp_audio = audio
    dur_vad, _, time_vad = run_vad(temp_audio, sampling_rate)
    duration = len(audio) / sampling_rate
    if dur_vad > 0.5 and not state.started_talking:
        print("started talking")
        state.started_talking = True
        return False
    print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")
    return (duration - dur_vad) > 1

def process_audio(audio:tuple, image: Image, state:AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream =  np.concatenate((state.stream, audio[1]))
    if image is None:
        state.image = {"file":""}
    else:
        state.image = {"file":str(image)}
    pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.pause_detected = pause_detected
    if state.pause_detected and state.started_talking:
        return gr.Audio(recording=False), state
    return None, state

def response(state:AppState = AppState()):
    max_new_tokens = 1024
    if not state.pause_detected and not state.started_talking:
        return None, AppState()
    audio_buffer = BytesIO()
    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")
    textin = ""
    with sr.AudioFile(audio_buffer) as source:
        audio_data=r.record(source)
        try:
            textin=r.recognize_google(audio_data,language='vi')
        except:
            textin = ""
        #state.conversation.append({"role": "user", "content": "Bạn: " + textin})   
    textout = ""
    if textin != "":
        print("Đang nghĩ...")
        state.message = {}
        state.message={"text": textin,"files": state.image["file"]}
        
        # phần phiên dịch
        txt = state.message["text"]
        messages= [] 
        images = []
        for i, msg in enumerate(state.history): 
            if isinstance(msg[0], tuple):
                messages.append({"role": "user", "content": [{"type": "text", "text": state.history[i][0]}, {"type": "image"}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": state.history[i][1]}]})
                images.append(Image.open(msg[0][0]).convert("RGB"))
            elif isinstance(state.history[i], tuple) and isinstance(msg[0], str):
                # messages are already handled
                pass
            elif isinstance(state.history[i][0], str) and isinstance(msg[0], str): # text only turn
                messages.append({"role": "user", "content": [{"type": "text", "text": msg[0]}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": msg[1]}]})

        # add current message
        if state.message["files"] != "": # examples
            image = Image.open(state.message["files"]).convert("RGB")
            images.append(image)
            messages.append({"role": "user", "content": [{"type": "text", "text": txt}, {"type": "image"}]})
        else: # regular input
            messages.append({"role": "user", "content": [{"type": "text", "text": txt}]})
        try:
            texts = processor.apply_chat_template(messages, add_generation_prompt=True)
            if images == []:
                inputs = processor(text=texts, return_tensors="pt").to("cuda")
            else:
                inputs = processor(text=texts, images=images, return_tensors="pt").to("cuda")
            streamer = TextIteratorStreamer(processor, skip_special_tokens=True, skip_prompt=True)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            buffer = ""
            for new_text in streamer:
                buffer += new_text
                time.sleep(0.01)
            state.textout=buffer
            textout=buffer
        except:
            print("Chưa lấy được thông tin dịch")
        if state.message["files"] != "":
            state.history.append([(txt,state.image["file"]),buffer])
            state.conversation.append({"role":"user","content":"Bạn: " + str(txt) + str(state.image["file"])})
            state.conversation.append({"role":"assistant", "content": "Bot: " + str(buffer)})
        else:
            state.history.append([txt,buffer])
            state.conversation.append({"role": "user", "content":"Bạn: " + str(txt)})
            state.conversation.append({"role": "assistant", "content":"Bot: " + str(buffer)})
    else:
        textout = "Tôi không nghe rõ"
    
    
    #phần đọc chữ đã dịch
    ssr = state.stream.tobytes()
    print("Đang đọc...")
    try:
        mp3 = gTTS(textout,tld='com.vn',lang='vi',slow=False)
        mp3_fp = BytesIO()
        mp3.write_to_fp(mp3_fp)
        srr=mp3_fp.getvalue()
    except:
        print("Lỗi không đọc được")
    finally:    
        mp3_fp.close()
    yield srr, AppState(conversation=state.conversation, history=state.history)

def start_recording_user(state:AppState):  # Sửa lỗi tại đây
    if not state.stopped:
        return gr.Audio(recording=True)

title = "vietnamese by tuphamkts"
description = "A vietnamese text-to-speech demo."

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Nói cho tôi nghe nào", sources="microphone", type="numpy")
            input_image = gr.Image(label="Hình ảnh của bạn", sources="upload", type="filepath")
        with gr.Column():
            chatbot = gr.Chatbot(label="Nội dung trò chuyện", type="messages")
            output_audio = gr.Audio(label="Trợ lý", autoplay=True)
    with gr.Row():
        output_image = gr.Image(label="Hình ảnh sau xử lý", sources="clipboard", type="filepath",visible=True)
    state = gr.State(value=AppState())
    stream = input_audio.stream(
        process_audio,
        [input_audio, input_image, state],
        [input_audio, state],
        stream_every=0.50,
        time_limit=30,
    )
    respond = input_audio.stop_recording(
        response,
        [state],
        [output_audio, state],
    )
    respond.then(lambda s: s.conversation, [state], [chatbot])
    respond.then(lambda s: s.image, [state], [output_image])
    restart = output_audio.stop(
        start_recording_user,
        [state, input_image],
        [input_audio],
    )
    cancel = gr.Button("Stop Conversation", variant="stop")
    cancel.click(lambda: (AppState(stopped=True), gr.Audio(recording=False)), None,
                [state, input_audio], cancels=[respond, restart])
demo.launch()