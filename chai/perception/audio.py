import speech_recognition as sr


class AudioInput:
    def __init__(self, language="en-US", timeout=5):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.timeout = timeout

    def listen_once(self):
        """Block until speech is detected, return transcript string."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[CHAI] Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=self.timeout)
                text = self.recognizer.recognize_google(audio, language=self.language)
                print(f"[STT] Heard: {text}")
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
