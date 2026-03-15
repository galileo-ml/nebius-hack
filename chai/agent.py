"""
LLM Agent for CHAI — replaces hardcoded state machine with an LLM decision loop.

Architecture: "Decide Async, Execute Sync"
  Physics loop (2400Hz)  → reads latest_decision → calls robot methods
  Agent loop   (~2Hz)    → reads latest_percept  → calls LLM → writes latest_decision
  VLM loops    (2Hz ea)  → reads camera frame    → calls VLM → writes latest_percept
"""

import json
import time
import threading
from collections import deque
from dataclasses import dataclass, field


VALID_ACTIONS = {"walk", "slow", "stop", "sweep_left", "sweep_right", "signal_clear", "signal_stop"}

SYSTEM_PROMPT = """You are the decision-making brain of a blind-guide robot (Unitree G1).
You walk ahead of a visually impaired user, detect obstacles, clear them with your arm, and narrate the environment.

Action vocabulary:
  walk         — walk forward at full speed (vx=0.20 m/s). Use when path is clear.
  slow         — slow forward walk (vx=0.10 m/s). Use when approaching an obstacle.
  stop         — halt completely. Use when obstacle is near or situation is unclear.
  sweep_right  — stop and sweep right arm to clear obstacle. Use when obstacle is directly in front.
  signal_clear — turn to face the human and wave to signal path is clear. Use after clearing an obstacle.
  signal_stop  — turn to face the human and hold up a hand to signal them to stop. Use BEFORE clearing a hazard.

Decision guidelines for obstacles:
  - If an obstacle is detected with distance "far" or "medium" (>1.0m): action=walk or slow (approach it)
  - If an obstacle is detected with distance "near" (<=1.0m):
      Step 1: action=signal_stop (if signal_stop_already_applied is false)
      Step 2: action=sweep_right (if signal_stop is done, but sweep_already_applied is false)
      Step 3: action=signal_clear (if sweep is done, but signal_clear_already_applied is false)
      Step 4: action=walk (if all the above are done)
  - ALWAYS check 'robot' context to see if an action is in progress. If action_in_progress != "none", output action=stop so it can finish!

Output ONLY valid JSON with this exact schema:
{"action": "walk|slow|stop|sweep_left|sweep_right|signal_clear|signal_stop", "vx": 0.20, "speech": "text or null", "reasoning": "brief"}
"""

_STALE_TIMEOUT = 5.0  # seconds before a decision is considered stale


@dataclass
class AgentDecision:
    action: str
    vx: float
    speech: str | None
    reasoning: str
    timestamp: float = field(default_factory=time.time)


_STOP_DECISION = AgentDecision(
    action="stop", vx=0.0, speech=None, reasoning="safety stop", timestamp=0.0
)


class RobotAgent:
    def __init__(self, client, llm_model: str, history_length: int = 6):
        self._client = client
        self._model = llm_model
        self._history: deque = deque(maxlen=history_length)

        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

        self._latest_decision: AgentDecision = AgentDecision(
            action="stop", vx=0.0, speech=None,
            reasoning="waiting for first perception", timestamp=time.time()
        )
        self._latest_percept: dict | None = None
        self._latest_context: dict | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._agent_loop, daemon=True, name="AgentLoop")
        self._thread.start()
        print("[AGENT] Agent loop started")

    def stop(self):
        self._running = False
        self._event.set()  # unblock the thread so it can exit

    def update_perception(self, percept: dict, robot_context: dict):
        """Called from physics loop each tick when perception is ready. Non-blocking."""
        with self._lock:
            self._latest_percept = percept
            self._latest_context = robot_context
        self._event.set()

    def get_decision(self, sim_dist: float) -> AgentDecision:
        """Thread-safe read of latest decision. Applies hard safety override."""
        with self._lock:
            decision = self._latest_decision

        # Stale guard: if decision is too old, force stop
        if time.time() - decision.timestamp > _STALE_TIMEOUT:
            return AgentDecision(
                action="stop", vx=0.0, speech=None,
                reasoning=f"stale decision ({_STALE_TIMEOUT}s timeout)",
                timestamp=decision.timestamp
            )

        # Hard safety override: never walk/slow into an obstacle
        if sim_dist < 1.0 and decision.action in ("walk", "slow"):
            return AgentDecision(
                action="stop", vx=0.0, speech=None,
                reasoning=f"safety override: sim_dist={sim_dist:.2f}m",
                timestamp=decision.timestamp
            )

        return decision

    def _agent_loop(self):
        while self._running:
            self._event.wait()
            self._event.clear()

            if not self._running:
                break

            with self._lock:
                percept = self._latest_percept
                context = self._latest_context

            if percept is None or context is None:
                continue

            try:
                new_decision = self._call_llm(percept, context)
                with self._lock:
                    self._latest_decision = new_decision
                print(f"[AGENT] {new_decision.action} (vx={new_decision.vx}) — {new_decision.reasoning}")
                if new_decision.speech:
                    print(f"[AGENT] speech: {new_decision.speech!r}")
            except Exception as e:
                print(f"[AGENT] LLM error (keeping previous decision): {e}")

    def _call_llm(self, percept: dict, context: dict) -> AgentDecision:
        user_content = json.dumps({
            "perception": percept,
            "robot": context,
        }, indent=2)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        # Add conversation history for continuity
        for h in self._history:
            messages.append(h)
        messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)

        # Update history
        self._history.append({"role": "user", "content": user_content})
        self._history.append({"role": "assistant", "content": raw})

        action = parsed.get("action", "stop")
        if action not in VALID_ACTIONS:
            print(f"[AGENT] Unknown action {action!r} — falling back to stop")
            action = "stop"

        vx = float(parsed.get("vx", 0.20 if action == "walk" else 0.10 if action == "slow" else 0.0))
        speech = parsed.get("speech") or None
        reasoning = parsed.get("reasoning", "")

        return AgentDecision(
            action=action,
            vx=vx,
            speech=speech,
            reasoning=reasoning,
            timestamp=time.time(),
        )
