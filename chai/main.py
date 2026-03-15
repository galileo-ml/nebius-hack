from openai import OpenAI

from perception.vlm_loop import PerceptionLoop
from perception.audio import AudioInput
from voice.tts import AudioOutput
from voice.translator import Translator
from robot.controller import RobotController
from robot.config import TOKEN_FACTORY_BASE_URL, TOKEN_FACTORY_API_KEY, LLM_MODEL
from state_machine import CHAI


def main():
    # Token Factory client (OpenAI-compatible)
    client = OpenAI(
        base_url=TOKEN_FACTORY_BASE_URL,
        api_key=TOKEN_FACTORY_API_KEY
    )

    robot      = RobotController()
    perception = PerceptionLoop(robot, client)
    audio_in   = AudioInput(language="hi-IN")
    audio_out  = AudioOutput()
    translator = Translator(client, LLM_MODEL)

    chai = CHAI(perception, audio_in, audio_out, translator, robot)

    perception.start()
    print("[CHAI] System ready.")
    chai.run()


if __name__ == "__main__":
    main()
