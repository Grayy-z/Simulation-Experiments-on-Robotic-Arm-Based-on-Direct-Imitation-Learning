import sys
import os
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.dirname(__file__))

from config.bc_config import BC_CONFIG
from src.utils.bc_trainer import BCTrainer
import matplotlib.pyplot as plt

def plot_training_history(history):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))

    # 训练损失
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # 验证损失
    plt.subplot(1, 3, 2)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)

    # 验证MSE
    plt.subplot(1, 3, 3)
    plt.plot(history['val_mse'], label='Validation MSE', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation MSE')
    plt.legend()
    plt.grid(True)

    # 保存路径
    save_dir = BC_CONFIG.get('models_dir', 'data/models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_history.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练历史图表已保存到: {save_path}")
    plt.show()


def train_dagger():
    """使用DAGGER算法训练"""
    print("=" * 50)
    print("开始DAGGER训练")
    print("=" * 50)

    from src.utils.dagger_trainer import DAGGER

    data_dir = os.path.join('data', 'cleaned_trajectories')
    if not os.path.exists(data_dir):
        print(f"轨迹数据目录不存在: {data_dir}")
        print("请先运行 clean_data.py 清洗数据")
        return

    # 创建DAGGER训练器
    dagger = DAGGER(BC_CONFIG, data_dir)

    final_policy = dagger.run()

    print(f"\nDAGGER训练完成！最终模型保存到: data/models/dagger_final_model.pth")


def train_with_data(self, states_list, actions_list):
    """使用内存中的数据训练模型（为DAGGER设计）"""
    print(f"使用内存数据进行训练: {len(states_list)} 个样本")

    class MemoryDataset(torch.utils.data.Dataset):
        def __init__(self, states, actions):
            self.states = states
            self.actions = actions

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return {
                'states': torch.FloatTensor(self.states[idx]),
                'action': torch.FloatTensor(self.actions[idx])
            }

    dataset = MemoryDataset(states_list, actions_list)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=self.config['batch_size'],
        shuffle=False
    )

    all_states = np.vstack(states_list)
    self.state_mean = all_states.mean(axis=0)
    self.state_std = all_states.std(axis=0)
    self.state_std[self.state_std < 1e-5] = 1.0

    print(f"数据统计: 均值={self.state_mean[:3]}, 标准差={self.state_std[:3]}")

    return self._train_with_loaders(train_loader, val_loader)


def _train_with_loaders(self, train_loader, val_loader):
    """使用数据加载器进行训练（内部方法）"""
    best_val_loss = float('inf')
    best_val_mse = float('inf')
    patience_counter = 0

    for epoch in range(self.config['num_epochs']):
        self.model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]}')
        for batch in pbar:
            states = batch['states'].to(self.device)
            actions = batch['action'].to(self.device)

            # 生成时间索引
            batch_size, seq_len, _ = states.shape
            time_indices = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

            # 前向传播
            mean, log_var = self.model(states, time_indices)

            # 计算损失
            loss, _, _, _ = self.compute_loss((mean, log_var), actions, states, time_indices)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = np.mean(train_losses)
        self.history['train_loss'].append(avg_train_loss)

        # 验证阶段
        avg_val_loss, avg_val_mse = self.validate(val_loader)
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_mse'].append(avg_val_mse)

        # 学习率调整
        self.scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train={avg_train_loss:.5f} | Val={avg_val_loss:.5f}")

        # 保存最佳模型
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_val_loss = avg_val_loss
            self.save_model(self.best_model_name)
            print(f"  保存最佳模型，验证MSE: {best_val_mse:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 提前停止
        if patience_counter >= self.config['patience']:
            print(f"  提前停止，验证损失在 {self.config['patience']} 轮内未改善")
            break

    self.save_model(self.final_model_name)

    return self.history

def main():
    """主训练函数 - 添加DAGGER选项"""
    print("\n=== 选择训练模式 ===")
    print("1. 标准行为克隆 (BC)")
    print("2. DAGGER算法训练")

    mode = input("请输入模式编号 (1/2): ").strip()

    print("=" * 50)
    print("开始训练行为克隆模型 (Loading .pkl data)")
    print("=" * 50)

    if mode == '2':
        train_dagger()
    else:

        # 轨迹数据目录
        data_dir = os.path.join('data', 'cleaned_trajectories')

        # 模型保存目录
        models_dir = BC_CONFIG['models_dir']
        os.makedirs(models_dir, exist_ok=True)

        if not os.path.exists(data_dir):
            print(f"轨迹数据目录不存在: {data_dir}")
            print("请先运行 clean_data.py 清洗数据")
            return

        print("\n=== 数据分析 ===")

        # 分析数据
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
            from test.analyze_trajectory_stats import analyze_trajectory_stats

            print("正在分析轨迹数据...")
            stats = analyze_trajectory_stats(data_dir)
            if stats and stats['num_trajectories'] > 0:
                print(f"\n分析结果:")
                print(f"  轨迹文件数量: {stats['num_trajectories']}")
                print(f"  总数据点数: {stats['total_points']}")
                print(f"  平均轨迹长度: {stats['avg_traj_length']:.2f}")
                print(f"  位置变化统计:")
                print(f"    最小值: {stats['pos_change_stats']['min']:.6f}")
                print(f"    最大值: {stats['pos_change_stats']['max']:.6f}")
                print(f"    均值: {stats['pos_change_stats']['mean']:.6f}")
                print(f"    标准差: {stats['pos_change_stats']['std']:.6f}")
            else:
                print("警告: 分析结果为空，可能未找到有效数据。")

        except Exception as e:
            print(f"数据分析时出错 (不影响训练): {e}")

        proceed = input("\n是否继续训练? (输入 y 继续，其他键取消): ").lower().strip()
        if proceed != 'y':
            print("训练取消")
            return

        traj_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        print(f"\n找到 {len(traj_files)} 个清洗后的轨迹文件 (.pkl)")

        if len(traj_files) == 0:
            print("没有找到 .pkl 轨迹数据文件")
            print("请检查 data/cleaned_trajectories 目录是否为空，或重新运行 clean_data.py")
            return

        # 创建训练器
        try:
            trainer = BCTrainer(BC_CONFIG, data_dir)
            history = trainer.train()
            plot_training_history(history)

            print("=" * 50)
            print("训练完成！")
            print(f"\n模型已保存到: {models_dir}/")
            print(f"  最佳模型: {BC_CONFIG['best_model_name']}")
            print(f"  最终模型: {BC_CONFIG['final_model_name']}")
            print("=" * 50)

        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()