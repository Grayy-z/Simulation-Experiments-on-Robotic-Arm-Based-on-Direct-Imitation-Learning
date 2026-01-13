import torch
import numpy as np
import os
import sys

class BCNetwork(torch.nn.Module):
    def __init__(self, input_dim=8, output_dim=8, hidden_size=128, num_layers=2,
                 dropout=0.2, time_encoding_dim=4):
        super().__init__()

        total_input_dim = input_dim + time_encoding_dim

        self.lstm = torch.nn.LSTM(
            input_size=total_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 均值分支
        self.mean_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, output_dim // 2)
        )

        # 方差分支
        self.var_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, output_dim // 2)
        )

        self.time_encoding_dim = time_encoding_dim

    def forward(self, x, time_indices=None):
        batch_size, seq_len, _ = x.shape

        # 生成时间编码
        if time_indices is None:
            time_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        # 时间编码函数
        def _time_encoding(indices):
            batch_size, seq_len = indices.shape
            indices = indices.float()

            freqs = torch.arange(self.time_encoding_dim // 2, device=indices.device).float()
            freqs = 10000 ** (-2 * freqs / self.time_encoding_dim)

            indices = indices.unsqueeze(-1)
            freqs = freqs.view(1, 1, -1)

            angles = indices * freqs
            sin_enc = torch.sin(angles)
            cos_enc = torch.cos(angles)

            time_enc = torch.zeros(batch_size, seq_len, self.time_encoding_dim, device=indices.device)
            time_enc[:, :, 0::2] = sin_enc
            time_enc[:, :, 1::2] = cos_enc

            return time_enc

        time_enc = _time_encoding(time_indices)
        x_augmented = torch.cat([x, time_enc], dim=-1)

        lstm_out, _ = self.lstm(x_augmented)
        last_output = lstm_out[:, -1, :]

        mean = self.mean_layers(last_output)
        log_var = self.var_layers(last_output)

        return mean, log_var

class BCController:
    def __init__(self, robot, env, model_path=None):
        self.robot = robot
        self.env = env
        
        if model_path is None:
            self.model_path = os.path.join('data', 'models', 'best_bc_model.pth')
        else:
            self.model_path = model_path

        self.model, self.state_mean, self.state_std, self.history_length = self.load_model()
        
        # 历史缓冲区
        self.history_length = 10 
        self.state_history = []
        
        # 初始状态
        self.current_pos = [0.5, 0, 0.3]
        self.current_orn = [1, 0, 0, 0]
        self.gripper_open = True
        
        self.init_state_history()
        print("BC控制器就绪。")

    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location='cpu')

        conf = checkpoint['config']

        # 检查是否为异方差模型
        is_heteroscedastic = checkpoint.get('heteroscedastic', False)

        if is_heteroscedastic:
            output_dim = conf.get('output_dim', 8)
        else:
            output_dim = conf.get('output_dim', 4)

        model = BCNetwork(
            input_dim=conf.get('input_dim', 8),
            output_dim=output_dim,
            hidden_size=conf.get('hidden_size', 128),
            num_layers=conf.get('num_layers', 2),
            dropout=conf.get('dropout', 0.2)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, np.array(checkpoint['state_mean']), np.array(checkpoint['state_std']), conf.get('seq_length', 25)

    def init_state_history(self):
        state = self.get_current_state()
        for _ in range(self.history_length):
            self.state_history.append(state)

    def get_current_state(self):
        grip = 1.0 if self.gripper_open else 0.0
        return np.array(self.current_pos + self.current_orn + [grip], dtype=np.float32)

    def update(self):
        curr_state = self.get_current_state()
        self.state_history.append(curr_state)
        self.state_history.pop(0)

        # 归一化
        seq = np.array(self.state_history)
        seq = (seq - self.state_mean) / self.state_std
        inp = torch.FloatTensor(seq).unsqueeze(0)


        with torch.no_grad():

            outputs = self.model(inp)

            print(f"DEBUG: model outputs type: {type(outputs)}")
            print(f"DEBUG: outputs length: {len(outputs) if isinstance(outputs, tuple) else 'not tuple'}")

            if isinstance(outputs, tuple):
                mean, log_var = outputs
                print(f"DEBUG: mean shape: {mean.shape}")
                print(f"DEBUG: log_var shape: {log_var.shape}")

                action = mean.numpy()[0]
            else:
                print(f"DEBUG: single output shape: {outputs.shape}")
                action = outputs.numpy()[0]


        delta_pos = action[:3]
        gripper_val = action[3]

        self.current_pos = [c + d for c, d in zip(self.current_pos, delta_pos)]

        self.current_pos[0] = max(0.2, min(1.0, self.current_pos[0]))
        self.current_pos[1] = max(-0.5, min(0.5, self.current_pos[1]))
        self.current_pos[2] = max(0.0, min(0.6, self.current_pos[2]))

        self.robot.move_to(self.current_pos, self.current_orn)

        # 夹爪控制
        if gripper_val > 0.5:
            self.gripper_open = True
        else:
            self.gripper_open = False

        self.robot.gripper_control(open=self.gripper_open)

        return False
        
    def reset(self):
        self.state_history = []
        self.init_state_history()