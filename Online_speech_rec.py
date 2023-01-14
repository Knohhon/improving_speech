import search_filler_words
import speech_recognition as sr
import time


# Распознавание речи с микрофона

# def mir_cer():
#    r = sr.Recognizer()
#    mic = sr.Microphone()
#    with mic as source:
#        r.adjust_for_ambient_noise(source, duration=0.5)
#        content = r.listen(source)
#    print(r.recognize_google(content,  language="ru-RU"))


def words_in_min(*args):
    mic_text_list = mic_text.lower().split(' ')
    return (len(mic_text_list) / tm_sec) * 60


startTime = time.perf_counter()

r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source, duration=0.5)
    content = r.listen(source)

endTime = time.perf_counter()

mic_text = r.recognize_google(content, language="ru-RU")
tm_sec = endTime - startTime


print(mic_text)
print(words_in_min(mic_text, tm_sec))

list_filler_words = search_filler_words.search_in_audiofile(mic_text)
print(list_filler_words)
