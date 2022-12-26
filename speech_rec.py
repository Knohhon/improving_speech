import speech_recognition

r = speech_recognition.Recognizer()
sample = speech_recognition.AudioFile("output.wav")
print(type(sample))

with sample as audio:
    content = r.record(audio)
    r.adjust_for_ambient_noise(audio, duration=0.5)
print(type(content))

print(r.recognize_google(content, language="ru-RU"))

