import speech_recognition as sr
r = sr.Recognizer()

mic = sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source)
    content = r.listen(source)

print(type(content))
print(r.recognize_google(content, language="ru-RU"))
