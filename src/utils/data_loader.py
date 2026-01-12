import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, seq_length=10, normalize=True):
        """
        Args:
            data_dir: 指向 'data/cleaned_trajectories'
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.normalize = normalize
        self.samples = []

        self._load_trajectories()

        if normalize:
            self._compute_normalization()

    def _load_trajectories(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"目录不存在: {self.data_dir}")

        # === 修改：只读取 .pkl 文件 ===
        traj_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        print(f"加载器: 找到 {len(traj_files)} 个清洗后的数据文件 (.pkl)")

        for file in traj_files:
            filepath = os.path.join(self.data_dir, file)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # clean_data.py 生成的格式是 {'states': ..., 'action': ...}
                self._process_trajectory(data)

    def _process_trajectory(self, data):
        states = data['states']
        actions = data['action']

        # 确保数据足够长
        if len(states) < self.seq_length + 1:
            return

        # 制作滑动窗口样本
        # 输入[t-seq : t] -> 输出 action[t]
        for i in range(len(states) - self.seq_length):
            state_seq = states[i : i + self.seq_length]
            action_target = actions[i + self.seq_length - 1] # 对应最后一个状态的动作

            self.samples.append({
                'state_sequence': state_seq,
                'action': action_target
            })

    def _compute_normalization(self):
        if len(self.samples) == 0:
            self.state_mean = np.zeros(8)
            self.state_std = np.ones(8)
            return

        # 只统计 State 的均值方差，Action 通常不归一化或单独处理
        all_states = [s['state_sequence'] for s in self.samples]
        all_states = np.vstack(all_states)

        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0)
        self.state_std[self.state_std < 1e-5] = 1.0  # 防止除零

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        state_seq = sample['state_sequence'].astype(np.float32)
        action = sample['action'].astype(np.float32)

        if self.normalize:
            state_seq = (state_seq - self.state_mean) / self.state_std

        return {
            'states': torch.FloatTensor(state_seq),
            'action': torch.FloatTensor(action)
        }

def create_data_loaders(data_dir, batch_size=32, train_split=0.8, seq_length=10):
    dataset = TrajectoryDataset(data_dir, seq_length=seq_length)
    
    if len(dataset) == 0:
        raise ValueError("数据集中没有样本！请检查 cleaned_trajectories 是否为空。")

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        dataset.state_mean,
        dataset.state_std
    )