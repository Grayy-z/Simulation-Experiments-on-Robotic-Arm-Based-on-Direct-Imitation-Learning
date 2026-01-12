import json
import os
import time

class TrajectoryRecorder:
    def __init__(self):
        self.data = []
        self.is_recording = False
        # 确保保存目录存在
        self.save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'trajectories')
        os.makedirs(self.save_dir, exist_ok=True)

    def start(self):
        self.data = []
        self.is_recording = True
        print(">>> 开始录制轨迹...")

    def stop(self):
        self.is_recording = False
        filename = f"traj_{int(time.time())}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f)
        print(f">>> 录制结束，已保存至: {filepath}")

    def record_step(self, pos, orn, gripper_open):
        """记录当前的一帧状态"""
        if self.is_recording:
            step_data = {
                "pos": pos,
                "orn": orn,
                "gripper_open": gripper_open
            }
            self.data.append(step_data)