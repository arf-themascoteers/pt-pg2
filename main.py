from scipy.io import wavfile
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

file_name = 'data/india.wav'
data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/framerate
print('Sample rate:',framerate,'Hz')
print('Total time:',len(sounddata)/framerate,'s')
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
input_audio, _ = librosa.load(file_name, sr=16000)
input_values = tokenizer(input_audio, return_tensors="pt").input_values
inter = model(input_values)
logits = inter.logits
predicted_ids = torch.argmax(logits, dim=-1)
batch = tokenizer.batch_decode(predicted_ids)
transcription = batch[0]
print(transcription)