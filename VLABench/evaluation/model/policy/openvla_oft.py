import collections
import base64

import numpy as np
import requests


class OpenVLAOFTPolicy:
    def __init__(self, host: str = "localhost", port: int = 8777, replan_steps: int = 4, timeout: float = 60.0) -> None:
        self.endpoint = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.timeout = timeout
        self.action_plan = collections.deque(maxlen=replan_steps)
        self.timestep = 0
        self.name = "openvla_oft"
        self.control_mode = "ee"

    def _build_payload(self, observation):
        rgb = observation["rgb"]
        return {
            "instruction": observation["instruction"],
            "left_image": np.asarray(rgb[0], dtype=np.uint8).tolist(),
            "full_image": np.asarray(rgb[2], dtype=np.uint8).tolist(),
            "wrist_image": np.asarray(rgb[3], dtype=np.uint8).tolist(),
            "state": np.asarray(observation["ee_state"], dtype=np.float32).tolist(),
        }

    def _request_action_chunk(self, observation):
        response = requests.post(self.endpoint, json=self._build_payload(observation), timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, list) or not result:
            raise RuntimeError(f"Unexpected OpenVLA-OFT response: {result}")
        return [self._decode_action(step) for step in result]

    @staticmethod
    def _decode_action(step):
        if isinstance(step, dict) and {"__numpy__", "dtype", "shape"}.issubset(step):
            buffer = base64.b64decode(step["__numpy__"])
            return np.frombuffer(buffer, dtype=np.dtype(step["dtype"])).reshape(step["shape"])

        return step

    def predict(self, observation, **kwargs):
        if self.timestep % self.replan_steps == 0 or not self.action_plan:
            action_chunk = self._request_action_chunk(observation)
            self.action_plan.extend(action_chunk[: self.replan_steps])

        self.timestep += 1
        raw_action = np.asarray(self.action_plan.popleft(), dtype=np.float32)
        if raw_action.shape[0] < 7:
            raise RuntimeError(f"OpenVLA-OFT action has invalid shape: {raw_action.shape}")

        robot_frame = np.asarray(observation.get("robot_frame", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        target_pos = raw_action[:3] + robot_frame
        target_euler = raw_action[3:6]
        gripper_open = bool(raw_action[-1] > 0.0)
        gripper_state = np.ones(2, dtype=np.float32) * 0.04 if gripper_open else np.zeros(2, dtype=np.float32)
        return target_pos, target_euler, gripper_state

    def reset(self) -> None:
        self.timestep = 0
        self.action_plan = collections.deque(maxlen=self.replan_steps)
