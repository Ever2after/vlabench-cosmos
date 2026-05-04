import collections

import numpy as np
import requests
from scipy.spatial.transform import Rotation


class MimicVideoPolicy:
    def __init__(self, host: str = "localhost", port: int = 8777, replan_steps: int = 4, timeout: float = 60.0) -> None:
        self.endpoint = f"http://{host}:{port}/act"
        self.reset_endpoint = f"http://{host}:{port}/reset"
        self.replan_steps = replan_steps
        self.timeout = timeout
        self.action_plan = collections.deque(maxlen=replan_steps)
        self.timestep = 0
        self.name = "mimic_video"
        self.control_mode = "ee"

    def _build_payload(self, observation):
        rgb = observation["rgb"]
        return {
            "instruction": observation["instruction"],
            "left_image": np.asarray(rgb[0], dtype=np.uint8).tolist(),
            "full_image": np.asarray(rgb[2], dtype=np.uint8).tolist(),
            "wrist_image": np.asarray(rgb[3], dtype=np.uint8).tolist(),
            "ee_state": np.asarray(observation["ee_state"], dtype=np.float32).tolist(),
            "robot_frame": np.asarray(observation["robot_frame"], dtype=np.float32).tolist(),
        }

    def _request_action_chunk(self, observation):
        response = requests.post(self.endpoint, json=self._build_payload(observation), timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, list) or not result:
            raise RuntimeError(f"Unexpected mimic-video response: {result}")
        return [np.asarray(step, dtype=np.float32) for step in result]

    def predict(self, observation, **kwargs):
        if self.timestep % self.replan_steps == 0 or not self.action_plan:
            action_chunk = self._request_action_chunk(observation)
            self.action_plan.extend(action_chunk[: self.replan_steps])

        self.timestep += 1
        raw_action = np.asarray(self.action_plan.popleft(), dtype=np.float32)
        if raw_action.shape[0] < 10:
            raise RuntimeError(f"mimic-video action has invalid shape: {raw_action.shape}")

        robot_frame = np.asarray(observation.get("robot_frame", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        target_pos = raw_action[:3] + robot_frame
        r1 = raw_action[3:6]
        r2 = raw_action[6:9]
        r1_norm = r1 / (np.linalg.norm(r1) + 1e-9)
        r2_orth = r2 - np.dot(r2, r1_norm) * r1_norm
        r2_norm = r2_orth / (np.linalg.norm(r2_orth) + 1e-9)
        r3 = np.cross(r1_norm, r2_norm)
        rot_matrix = np.stack([r1_norm, r2_norm, r3], axis=0)
        target_euler = Rotation.from_matrix(rot_matrix).as_euler("xyz").astype(np.float32)
        gripper_state = np.ones(2, dtype=np.float32) * 0.04 if raw_action[9] > 0.02 else np.zeros(2, dtype=np.float32)
        return target_pos.astype(np.float32), target_euler, gripper_state

    def reset(self) -> None:
        self.timestep = 0
        self.action_plan = collections.deque(maxlen=self.replan_steps)
        try:
            requests.post(self.reset_endpoint, timeout=min(self.timeout, 5.0)).raise_for_status()
        except requests.RequestException:
            pass