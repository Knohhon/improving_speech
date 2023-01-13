import pyaudio
import wave

# Запись звука в аудио файл

CHUNK = 1024
FRT = pyaudio.paInt16
CHAN = 1
RT = 44100
REC_SEC = 20
OUTPUT = "output.wav"
p = pyaudio.PyAudio()

# Start audio stream
audio_stream = p.open(format=FRT, channels=CHAN, rate=RT, input=True, frames_per_buffer=CHUNK)
print("rec")
frames = []
for i in range(0, int(RT / CHUNK * REC_SEC)):
    data = audio_stream.read(CHUNK)
    frames.append(data)
print("done")

audio_stream.stop_stream()
audio_stream.close()
p.terminate()

# Recording audio to audio file
w = wave.open(OUTPUT, 'wb')
w.setnchannels(CHAN)
w.setsampwidth(p.get_sample_size(FRT))
w.setframerate(RT)
w.writeframes(b''.join(frames))
w.close()
