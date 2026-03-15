from enum import Enum, auto

from voice.phrases import GREETING_EN, RESUME_EN, OBSTACLE_WARNINGS


class State(Enum):
    IDLE              = auto()
    HUMAN_DETECTED    = auto()
    GREETING          = auto()
    AWAIT_RESPONSE    = auto()
    TRANSLATING       = auto()
    GUIDING           = auto()
    OBSTACLE_DETECTED = auto()
    WARNING           = auto()
    CLEARING          = auto()
    RESUMING          = auto()


class CHAI:
    def __init__(self, perception, audio_in, audio_out, translator, robot):
        self.perception  = perception
        self.audio_in    = audio_in
        self.audio_out   = audio_out
        self.translator  = translator
        self.robot       = robot
        self.state       = State.IDLE
        self.warned_obstacles = set()  # tracks obstacles already warned about

        self._hindi_response    = None
        self._current_obstacle  = None

    def _obstacle_key(self, obs):
        """Deduplicate warnings by position+type."""
        return f"{obs.get('type')}_{obs.get('side')}"

    def run(self):
        print("[CHAI] Starting main loop...")
        while True:
            percept = self.perception.get()
            self._tick(percept)

    def _tick(self, percept):
        human    = percept["human"]
        obstacle = percept["obstacle"]

        # --- IDLE ---
        if self.state == State.IDLE:
            if human.get("human") and human.get("approaching"):
                self.state = State.HUMAN_DETECTED

        # --- HUMAN DETECTED ---
        elif self.state == State.HUMAN_DETECTED:
            self.robot.stop()
            self.state = State.GREETING

        # --- GREETING ---
        elif self.state == State.GREETING:
            self.audio_out.speak(GREETING_EN, lang="en")
            self.state = State.AWAIT_RESPONSE

        # --- AWAIT RESPONSE ---
        elif self.state == State.AWAIT_RESPONSE:
            spoken = self.audio_in.listen_once()
            if spoken:
                self._hindi_response = spoken
                self.state = State.TRANSLATING
            else:
                # No response heard — proceed to guiding
                self.state = State.GUIDING

        # --- TRANSLATING ---
        elif self.state == State.TRANSLATING:
            translation  = self.translator.hindi_to_english(self._hindi_response)
            announcement = f"The person said: {translation}"
            print(f"[Translation] {announcement}")
            self.audio_out.speak(announcement, lang="en")
            self.state = State.GUIDING

        # --- GUIDING ---
        elif self.state == State.GUIDING:
            self.robot.walk_forward()
            if obstacle.get("obstacle"):
                key = self._obstacle_key(obstacle)
                if key not in self.warned_obstacles:
                    self.warned_obstacles.add(key)
                    self._current_obstacle = obstacle
                    self.state = State.OBSTACLE_DETECTED

        # --- OBSTACLE DETECTED ---
        elif self.state == State.OBSTACLE_DETECTED:
            self.robot.stop()
            self.state = State.WARNING

        # --- WARNING ---
        elif self.state == State.WARNING:
            side         = self._current_obstacle.get("side", "center")
            warning_text = OBSTACLE_WARNINGS["en"].get(side, OBSTACLE_WARNINGS["en"]["center"])
            self.audio_out.speak(warning_text, lang="en")
            self.state = State.CLEARING

        # --- CLEARING ---
        elif self.state == State.CLEARING:
            side = self._current_obstacle.get("side", "center")
            if side == "left":
                self.robot.arm.clear_left()
            else:
                self.robot.arm.clear_right()
            self.state = State.RESUMING

        # --- RESUMING ---
        elif self.state == State.RESUMING:
            self.audio_out.speak(RESUME_EN, lang="en")
            self.state = State.GUIDING
