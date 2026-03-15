class Translator:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def hindi_to_english(self, hindi_text):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a translator. The user will give you Hindi text. "
                    "Translate it naturally to English. "
                    "Return ONLY the English translation, nothing else."
                )},
                {"role": "user", "content": hindi_text}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
