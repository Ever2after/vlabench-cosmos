import collections

import numpy as np
import requests


class CosmosPolicy:
    def __init__(self, host: str = "localhost", port: int = 8777, replan_steps: int = 4, timeout: float = 60.0) -> None:
        self.endpoint = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.timeout = timeout
        self.action_plan = collections.deque(maxlen=replan_steps)
        self.timestep = 0
        self.name = "cosmos_policy"
        self.control_mode = "ee"

    def _request_action_chunk(self, observation):
        rgb = observation["rgb"]
        payload = {
            "task_description": observation["instruction"],
            "primary_image": np.asarray(rgb[2], dtype=np.uint8).tolist(),
            "wrist_image": np.asarray(rgb[3], dtype=np.uint8).tolist(),
            "proprio": np.asarray(observation["ee_state"], dtype=np.float32).tolist(),
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or "actions" not in result:
            raise RuntimeError(f"Unexpected Cosmos Policy response: {result}")
        return result["actions"]

    def predict(self, observation, **kwargs):
        if self.timestep % self.replan_steps == 0 or not self.action_plan:
            action_chunk = self._request_action_chunk(observation)
            if not action_chunk:
                raise RuntimeError("Cosmos Policy server returned an empty action chunk")
            self.action_plan.extend(action_chunk[: self.replan_steps])

        self.timestep += 1
        raw_action = np.asarray(self.action_plan.popleft(), dtype=np.float32)
        if raw_action.shape[0] < 7:
            raise RuntimeError(f"Cosmos Policy action has invalid shape: {raw_action.shape}")

        robot_frame = np.asarray(observation.get("robot_frame", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        target_pos = raw_action[:3] + robot_frame
        target_euler = raw_action[3:6]
        gripper_open = float(np.mean(raw_action[-2:])) > 0.03 if raw_action.shape[0] >= 8 else raw_action[-1] > 0.03
        gripper_state = np.ones(2, dtype=np.float32) * 0.04 if gripper_open else np.zeros(2, dtype=np.float32)
        return target_pos, target_euler, gripper_state

    def reset(self) -> None:
        self.timestep = 0
        self.action_plan = collections.deque(maxlen=self.replan_steps)