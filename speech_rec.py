import speech_recognition

# Распознавание речи из аудио файла

filename = 'output.wav'


def audio_rec(filename):
    r = speech_recognition.Recognizer()
    sample = speech_recognition.AudioFile(filename)

    with sample as audio:
        content = r.record(audio)
        r.adjust_for_ambient_noise(audio, duration=0.5)

    text = r.recognize_google(content, language="ru-RU")
    return text


finaltext = audio_rec(filename).lower()
print(finaltext)
