import speech_recognition
import wave
import contextlib

# Распознавание речи из аудио файла

filename = 'output.wav'


def words_in_min_audiofile(fname, ftext):
    ts = len_audiofile(fname)
    ftext_list = ftext.split()
    return (len(ftext_list) / ts) * 60


def len_audiofile(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def audio_rec(fname):
    r = speech_recognition.Recognizer()
    sample = speech_recognition.AudioFile(fname)

    with sample as audio:
        content = r.record(audio)
        r.adjust_for_ambient_noise(audio, duration=0.5)

    text = r.recognize_google(content, language="ru-RU")
    return text


finaltext = audio_rec(filename).lower()
score = words_in_min_audiofile(filename, finaltext)

print(finaltext)
print(score)
