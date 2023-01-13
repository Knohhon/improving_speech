import speech_recognition as sr

# Распознавание речи с микрофона

# def mir_cer():
#    r = sr.Recognizer()
#    mic = sr.Microphone()
#    with mic as source:
#        r.adjust_for_ambient_noise(source, duration=0.5)
#        content = r.listen(source)
#    print(r.recognize_google(content,  language="ru-RU"))


r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source, duration=0.5)
    content = r.listen(source)
print(r.recognize_google(content, language="ru-RU"))


