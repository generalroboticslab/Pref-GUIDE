import time
import os
import wave

import openai
import pyaudio
import whisper
import torch
import numpy as np


# import pathlib

# whisper_model = whisper.load_model("base")

# SYSTEM_PROMPT = """You always respond in the same format as the examples below."""

# USER_PROMPT = """Your task is to map natural language instructions into scores between
# 0 to 1 for a list of given actions
# [forward, backward, turn left, turn right, pick up]. \n
# Example1:\nInstruction:Do not turn left.\n
# Scores:\nfoward:1\nbackward:1\nturn left:0\nturn right:1\npick up:0\n\n
# Example2:\nInstruction:Rotate without moving.\n
# cores:\nforward:0\nbackward:0\nturn left:1\nturn right:1\npick up:0\nLet's start!
# \nInstruction: """


def audio2reward(audio_file_path):
    instruction = transcribe_audio(audio_file_path)
    if instruction == "":
        return [0, 0, 0, 0, 0]
    try:
        reward_dict = text2reward(instruction)
        print('Audio transcription:', instruction)
        print('Audio Reward: ', [round(r, 5) for r in reward_dict])
    except:
        reward_dict = [0, 0, 0, 0, 0]
    # time.sleep(1.6)
    # reward_dict = [0.5, 0.2, 0.1, 0.1, 0.1]
    return reward_dict


def transcribe_audio(audio_file_path):
    
    result = whisper_model.transcribe(audio_file_path)
    return result["text"]
    # openai.api_key = "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"
    # audio_file = open(audio_file_path, "rb")
    # transcript = openai.Audio.transcribe("whisper-1", audio_file)
    # return transcript["text"]


def get_gpt_response(user_message, system_message=""):
    print("GPT process: ", os.getpid())
    openai.api_key = "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"
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    output = completion.choices[0].message.content
    return output


def text2reward(instruction):
    llm_out = get_gpt_response(USER_PROMPT + instruction, SYSTEM_PROMPT)
    for line in llm_out.split("\n"):
        if line.startswith("forward"):
            forward = float(line[line.index(":") + 1 :])
        if line.startswith("backward"):
            backward = float(line[line.index(":") + 1 :])
        if line.startswith("turn left"):
            left = float(line[line.index(":") + 1 :])
        if line.startswith("turn right"):
            right = float(line[line.index(":") + 1 :])
        if line.startswith("pick up"):
            pick = float(line[line.index(":") + 1 :])

    total = forward + backward + left + right + pick + 5e-3

    forward = (forward + 1e-3) / total
    backward = (backward + 1e-3) / total
    left = (left + 1e-3) / total
    right = (right + 1e-3) / total
    pick = (pick + 1e-3) / total

    reward_list = [forward, backward, left, right, pick]
    if max(reward_list) - min(reward_list) < 0.1:
        reward_list = [0, 0, 0, 0, 0]

    return reward_list
    # {"forward": forward, "backward": backward, "left": left, "right": right}


class Audio_Streamer:
    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.buffer = []

    def start_streaming(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )

    def get_sample(self):
        data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
        self.buffer.append(data)
        # return data

    def stop_streaming(self):
        self.stream.stop_streaming()
        self.stream.close()
        self.audio.terminate()

    def save_to_file(self, file_name):
        # pathlib.Path('audio').mkdir(parents=True, exist_ok=True)
        sound_file = wave.open(
            "crew_algorithms/ddpg/audio/%s.wav" % file_name, "wb"
        )
        sound_file.setnchannels(1)
        sound_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(self.buffer))
        sound_file.close()
        self.buffer = []

    # def to_torch_tensor(self):

    #     final_buffer = []

    #     for data in self.buffer:
    #         # convert each small buffer to numpy array
    #         npbuffer = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            
    #         # normalize between -1 and 1
    #         npbuffer = npbuffer / 32768

    #         # reshape to time-major frame
    #         npbuffer = np.reshape(npbuffer, (-1, 1))

    #         final_buffer.append(npbuffer)  # append to final buffer

    #     # concatenate all the small buffers
    #     final_buffer_np = np.concatenate(final_buffer)

    #     # convert final numpy array to torch tensor
    #     torchbuffer = torch.from_numpy(final_buffer_np).to('cuda')

    #     self.buffer = []

        # return torchbuffer


if __name__ == "__main__":
    ins = "Keep going"
    print(text2reward(ins))

# s = start_streaming()
# data = []

# for i in range(100):
#     #time.sleep(0.5)
#     d = s.read(1024)
#     data.append(d)
#     #print(d[:5])

# sound_file = wave.open("myrecording.wav", 'wb')
# sound_file.setnchannels(1)
# sound_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
# sound_file.setframerate(44100)
# sound_file.writeframes(b''.join(data))
# sound_file.close()
# print(data)
