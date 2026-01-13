import torch
import numpy as np
import os
from .bc_controller import BCController, BCNetwork


class DAGGERController(BCController):
    """基于DAGGER训练的控制器的推理版本"""

    def __init__(self, robot, env, model_path=None):
        super().__init__(robot, env, model_path)

        self.use_uncertainty = True
        self.uncertainty_threshold = 0.5

        print("DAGGER控制器就绪（带不确定性估计）")

    def get_action_with_uncertainty(self, seq):
        """获取动作并返回不确定性"""
        with torch.no_grad():
            mean, log_var = self.model(seq)
            # 计算方差
            variance = torch.exp(log_var)

            # 平均不确定性
            avg_uncertainty = variance.mean().item()

            # 采样动作
            if self.use_uncertainty and avg_uncertainty > self.uncertainty_threshold:
                # 添加探索噪声
                noise = torch.randn_like(mean) * torch.sqrt(variance) * 0.1
                action = mean + noise
            else:
                action = mean

            return action.numpy()[0], avg_uncertainty

    def update(self):
        curr_state = self.get_current_state()
        self.state_history.append(curr_state)
        self.state_history.pop(0)

        seq = np.array(self.state_history)
        seq = (seq - self.state_mean) / self.state_std
        inp = torch.FloatTensor(seq).unsqueeze(0)

        action, uncertainty = self.get_action_with_uncertainty(inp)

        if uncertainty > self.uncertainty_threshold:
            print(f"高不确定性: {uncertainty:.3f}，采取保守动作")
            action[:3] *= 0.5

        delta_pos = action[:3]
        gripper_val = action[3]
        self.current_pos = [c + d for c, d in zip(self.current_pos, delta_pos)]

        self.current_pos[0] = max(0.2, min(1.0, self.current_pos[0]))
        self.current_pos[1] = max(-0.5, min(0.5, self.current_pos[1]))
        self.current_pos[2] = max(-0.02, min(0.6, self.current_pos[2]))

        self.robot.move_to(self.current_pos, self.current_orn)

        if gripper_val > 0.5:
            self.gripper_open = True
        else:
            self.gripper_open = False

        self.robot.gripper_control(open=self.gripper_open)

        return False, uncertainty  # 返回不确定性用于监控