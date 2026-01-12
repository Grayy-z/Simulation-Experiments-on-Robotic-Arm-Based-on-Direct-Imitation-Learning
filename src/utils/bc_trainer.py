import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F


class BCNetwork(nn.Module):
    def __init__(self, input_dim=8, output_dim=8, hidden_size=128, num_layers=2,
                 dropout=0.2, time_encoding_dim=4):
        super().__init__()

        # 总输入维度 = 状态维度 + 时间编码维度
        total_input_dim = input_dim + time_encoding_dim

        # LSTM编码器（接收增强的输入）
        self.lstm = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 均值分支
        self.mean_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim // 2)  # 输出均值（4维）
        )

        # 方差分支（增强版：时序感知）
        self.var_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 添加额外的时序处理层
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim // 2)  # 输出log方差（4维）
        )

        # 时间编码器（新增）
        self.time_encoding_dim = time_encoding_dim

    def forward(self, x, time_indices=None):
        # x形状: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # 生成时间编码（如果没有提供）
        if time_indices is None:
            # 默认：序列中每个位置的时间步
            time_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        # 简单的时间编码：正弦+余弦编码
        time_enc = self._time_encoding(time_indices)  # (batch, seq_len, time_encoding_dim)

        # 拼接状态和时间编码
        x_augmented = torch.cat([x, time_enc], dim=-1)

        # LSTM处理
        lstm_out, _ = self.lstm(x_augmented)
        last_output = lstm_out[:, -1, :]

        # 输出均值和log方差
        mean = self.mean_layers(last_output)
        log_var = self.var_layers(last_output)

        return mean, log_var

    def _time_encoding(self, indices):
        """正弦位置编码（类似Transformer）"""
        # indices形状: (batch, seq_len)
        batch_size, seq_len = indices.shape
        indices = indices.float()

        # 生成频率
        freqs = torch.arange(self.time_encoding_dim // 2, device=indices.device).float()
        freqs = 10000 ** (-2 * freqs / self.time_encoding_dim)

        # 计算正弦和余弦编码
        indices = indices.unsqueeze(-1)  # (batch, seq_len, 1)
        freqs = freqs.view(1, 1, -1)  # (1, 1, time_encoding_dim//2)

        angles = indices * freqs
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        # 交错拼接正弦和余弦
        time_enc = torch.zeros(batch_size, seq_len, self.time_encoding_dim, device=indices.device)
        time_enc[:, :, 0::2] = sin_enc
        time_enc[:, :, 1::2] = cos_enc

        return time_enc

class BCTrainer:
    """行为克隆训练器"""

    def __init__(self, config, data_dir):
        self.config = config
        self.data_dir = data_dir
        self.device = torch.device(config['device'])

        # 初始化模型（传递时间编码维度）
        self.model = BCNetwork(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            time_encoding_dim=config.get('time_encoding_dim', 4)  # 新增
        ).to(self.device)

        print(f"模型结构:")
        print(f"  输入维度: {config['input_dim']}")
        print(f"  输出维度: {config['output_dim']}")
        print(f"  隐藏层大小: {config['hidden_size']}")
        print(f"  LSTM层数: {config['num_layers']}")
        print(f"  Dropout: {config['dropout']}")

        # 基础损失函数
        self.criterion = nn.MSELoss()

        # 读取权重配置 (默认位置权重1.0, 夹爪权重1.0)
        self.weights = config.get('loss_weights', {'pos': 1.0, 'gripper': 1.0})
        print(f"Loss权重: Position={self.weights['pos']}, Gripper={self.weights['gripper']}")

        # 从配置读取方向感知权重
        self.direction_weight = config.get('direction_weight', 0.5)
        self.variance_peak_weight = config.get('variance_peak_weight', 1.0)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5  # 增加L2正则化，防止过拟合
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': []
        }

        # 设置模型保存路径
        self.models_dir = config.get('models_dir', os.path.join('data', 'models'))
        self.best_model_name = config.get('best_model_name', 'best_bc_model.pth')
        self.final_model_name = config.get('final_model_name', 'final_bc_model.pth')

        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)

    def compute_loss(self, pred_action, target_action, states=None, time_indices=None):
        """稳定的异方差损失函数"""
        mean, log_var = pred_action

        # === 修复1: 确保方差不会过小 ===
        # 添加一个小的epsilon防止数值不稳定
        log_var = torch.clamp(log_var, min=-10, max=10)  # 限制log_var范围
        variance = torch.exp(log_var) + 1e-6  # 防止除零

        # === 修复2: 更稳定的NLL计算 ===
        # NLL = 0.5 * [log_var + (y-μ)²/σ² + log(2π)]
        # 添加常数项使损失为正
        const_term = 0.5 * np.log(2 * np.pi)

        # 逐元素计算损失，更稳定
        nll = 0.5 * log_var + (target_action - mean) ** 2 / (2 * variance) + const_term

        # === 修复3: 更保守的权重 ===
        # 为位置和夹爪设置不同的权重
        pos_weights = torch.tensor([
            self.weights['pos'],  # x
            self.weights['pos'],  # y
            self.weights.get('pos_z', 2.0),  # z（降低权重）
            self.weights['gripper']  # gripper
        ]).to(self.device)

        # 扩展权重
        pos_weights = pos_weights.unsqueeze(0).expand_as(nll)
        weighted_nll = nll * pos_weights

        # 总损失
        total_loss = weighted_nll.mean()

        # === 修复4: 添加L2正则化 ===
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2)
        total_loss = total_loss + 1e-4 * l2_reg

        # 监控指标
        with torch.no_grad():
            mse = ((mean - target_action) ** 2).mean(dim=0)
            pos_loss = mse[:3].mean().item()
            grip_loss = mse[3].item()
            z_loss = mse[2].item()

        return total_loss, pos_loss, grip_loss, z_loss

    def _compute_direction_aware_loss(self, pred_actions, target_actions, states):
        """计算方向感知损失：鼓励模型在方向变化时提高不确定性"""
        # 从状态中提取当前位置（假设states的最后一维是[xyz+orn+grip]）
        # 我们只需要位置信息（前3维）
        current_positions = states[:, -1, :3]  # (batch, 3)

        # 预测的下一个位置
        next_pos_pred = current_positions + pred_actions[:, :3]
        next_pos_target = current_positions + target_actions[:, :3]

        # 计算速度向量（当前到预测/目标）
        pred_velocity = pred_actions[:, :3]  # 预测的速度
        target_velocity = target_actions[:, :3]  # 目标的速度

        # 计算方向一致性损失（余弦相似度）
        # 当预测方向和目标方向相反时，损失更大
        cos_sim = F.cosine_similarity(pred_velocity, target_velocity, dim=-1)  # (batch,)

        # 方向错误：当cos_sim为负时，表示方向相反
        direction_error = (1 - cos_sim) / 2  # 归一化到[0, 1]，0表示同向，1表示反向

        return direction_error.mean()

    def _compute_variance_peak_loss(self, log_var, time_indices):
        """鼓励模型在序列中间部分（通常是转折点）提高方差"""
        # time_indices形状: (batch, seq_len)
        # 我们只关心最后一个时间步（因为是单步预测）
        if time_indices.dim() == 2:
            time_positions = time_indices[:, -1]  # (batch,)
        else:
            time_positions = time_indices

        # 假设序列长度固定（比如25），我们鼓励在中间时间步（10-15）有更高的方差
        # 计算一个钟形权重：中间高，两端低
        seq_length = self.config.get('seq_length', 25)
        center = seq_length // 2
        width = seq_length // 4

        # 计算每个时间步的理想方差权重
        ideal_weights = torch.exp(-(time_positions - center) ** 2 / (2 * width ** 2))

        # 我们实际想要的是log_var与ideal_weights正相关
        # 当前log_var的均值（跨动作维度）
        mean_log_var = log_var.mean(dim=-1)  # (batch,)

        # 我们希望mean_log_var与ideal_weights负相关（因为ideal_weights在中间高）
        # 但实际上我们希望方差在中间高，所以需要调整
        variance_peak_loss = - (mean_log_var * ideal_weights).mean()

        return variance_peak_loss

    def train(self):
        """训练BC模型"""
        print(f"\n使用设备: {self.device}")
        print(f"训练配置:")
        print(f"  批次大小: {self.config['batch_size']}")
        print(f"  学习率: {self.config['learning_rate']}")
        print(f"  训练轮数: {self.config['num_epochs']}")
        print(f"  序列长度: {self.config['seq_length']}")

        # 创建数据加载器
        # 注意: 这里需要在 utils 目录下有 data_loader.py 文件，并包含 create_data_loaders 函数
        from .data_loader import create_data_loaders

        train_loader, val_loader, self.state_mean, self.state_std = create_data_loaders(
            self.data_dir,
            batch_size=self.config['batch_size'],
            train_split=self.config['train_split'],
            seq_length=self.config['seq_length']
        )

        print(f"\n数据统计:")
        print(f"  训练样本数: {len(train_loader.dataset)}")
        print(f"  验证样本数: {len(val_loader.dataset)}")
        print(f"  状态均值: {self.state_mean}")
        print(f"  状态标准差: {self.state_std}")

        best_val_loss = float('inf')
        best_val_mse = float('inf')
        patience_counter = 0

        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            self.model.train()
            train_losses = []
            pos_losses = []
            grip_losses = []
            z_losses = []

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]}')
            for batch in pbar:
                states = batch['states'].to(self.device)
                actions = batch['action'].to(self.device)

                # 生成时间索引（用于时间编码）
                batch_size, seq_len, _ = states.shape
                time_indices = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

                # 前向传播（传递时间索引）
                mean, log_var = self.model(states, time_indices)

                # 计算增强的损失（传递状态和时间索引）
                loss, l_pos, l_grip, l_z = self.compute_loss(
                    (mean, log_var), actions, states, time_indices
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 记录损失
                train_losses.append(loss.item())
                pos_losses.append(l_pos)
                grip_losses.append(l_grip)
                z_losses.append(l_z)

                pbar.set_postfix({
                    'L_Tot': f'{loss.item():.4f}',
                    'L_Z': f'{l_z:.4f}',
                    'L_Grip': f'{l_grip:.4f}'
                })

            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)

            # 验证阶段
            # 验证阶段
            avg_val_loss, avg_val_mse = self.validate(val_loader)  # 修改：返回两个值
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_mse'].append(avg_val_mse)

            # 学习率调整
            self.scheduler.step(avg_val_loss)

            print(f"Epoch {epoch + 1}: Train={avg_train_loss:.5f} (Pos={np.mean(pos_losses):.5f}, Grip={np.mean(grip_losses):.5f}) | Val={avg_val_loss:.5f}")

            # 保存最佳模型 - 使用MSE而不是NLL
            if avg_val_mse < best_val_mse:  # 修改：使用MSE
                best_val_mse = avg_val_mse
                best_val_loss = avg_val_loss  # 也记录一下NLL
                self.save_model(self.best_model_name)
                print(f"  保存最佳模型，验证MSE: {best_val_mse:.6f} (NLL: {avg_val_loss:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 提前停止
            if patience_counter >= self.config['patience']:
                print(f"  提前停止，验证损失在 {self.config['patience']} 轮内未改善")
                break

        # 保存最终模型
        self.save_model(self.final_model_name)

        return self.history

    def validate(self, val_loader):
        """验证模型 - 返回NLL损失和MSE"""
        self.model.eval()
        val_losses = []
        val_mses = []  # 新增：记录MSE

        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(self.device)
                actions = batch['action'].to(self.device)

                # 前向传播
                mean, log_var = self.model(states)

                # 计算NLL损失
                loss, _, _, _ = self.compute_loss((mean, log_var), actions)
                val_losses.append(loss.item())

                # 计算MSE（用于模型选择）
                mse = ((mean - actions) ** 2).mean().item()
                val_mses.append(mse)

        return np.mean(val_losses), np.mean(val_mses)  # 返回两个值

    def save_model(self, filename):
        """保存模型和标准化参数"""
        save_path = os.path.join(self.models_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'state_mean': self.state_mean.tolist(),
            'state_std': self.state_std.tolist(),
            'history': self.history,
            'heteroscedastic': True
        }, save_path)
        print(f"模型已保存到: {save_path}")

    @staticmethod
    def load_model(filename):
        """加载训练好的模型"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"模型文件不存在: {filename}")

        checkpoint = torch.load(filename, map_location='cpu')

        # 创建模型
        model = BCNetwork(
            input_dim=checkpoint['config']['input_dim'],
            output_dim=checkpoint['config']['output_dim'],
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers'],
            dropout=checkpoint['config']['dropout']
        )

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])

        return model, checkpoint['state_mean'], checkpoint['state_std']