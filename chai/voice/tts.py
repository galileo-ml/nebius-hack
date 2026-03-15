from gtts import gTTS
import playsound
import tempfile
import os


class AudioOutput:
    def speak(self, text, lang="en"):
        """Speak text aloud. lang='en' or 'hi'."""
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        tts.save(path)
        playsound.playsound(path)
        os.unlink(path)
