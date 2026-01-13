import torch
import numpy as np
import os
from tqdm import tqdm
import pickle
import time


class DAGGER:
    """简化版DAGGER算法实现"""

    def __init__(self, config, expert_data_dir):
        self.config = config
        self.expert_data_dir = expert_data_dir
        self.device = torch.device(config['device'])

        # DAGGER参数
        self.num_iterations = config.get('dagger_iterations', 5)
        self.samples_per_iter = config.get('dagger_samples_per_iter', 10)
        self.use_beta_schedule = config.get('use_beta_schedule', True)

        # 数据存储
        self.aggregated_data = {
            'states': [],
            'actions': []
        }

        # 创建保存目录
        self.dagger_dir = os.path.join('data', 'dagger')
        os.makedirs(self.dagger_dir, exist_ok=True)

        print(f"DAGGER配置:")
        print(f"  迭代次数: {self.num_iterations}")
        print(f"  每轮样本: {self.samples_per_iter}")

    def load_expert_data(self):
        """加载初始专家数据"""
        print("加载专家数据...")

        if not os.path.exists(self.expert_data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.expert_data_dir}")

        traj_files = [f for f in os.listdir(self.expert_data_dir)
                      if f.endswith('.pkl')]

        if len(traj_files) == 0:
            raise ValueError("没有找到轨迹数据文件 (.pkl)")

        for file in traj_files[:5]:
            filepath = os.path.join(self.expert_data_dir, file)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

                self.aggregated_data['states'].extend(data['states'].tolist())
                self.aggregated_data['actions'].extend(data['action'].tolist())

        print(f"已加载 {len(self.aggregated_data['states'])} 个专家样本")

    def get_beta(self, iteration):
        """获取β值（专家使用概率）"""
        if not self.use_beta_schedule:
            return 0.0

        p = 0.7
        beta = p ** (iteration - 1)
        return min(beta, 1.0)

    def collect_synthetic_data(self, iteration, policy_path=None):
        """收集合成数据（简化版）"""
        print(f"迭代 {iteration}: 生成合成数据...")

        beta = self.get_beta(iteration)
        print(f"  β值: {beta:.3f} (使用专家概率)")

        synthetic_states = []
        synthetic_actions = []

        num_samples = self.samples_per_iter * 10

        for i in range(num_samples):
            state = np.random.randn(8).astype(np.float32)
            state[0] = np.random.uniform(0.3, 0.7)
            state[1] = np.random.uniform(-0.3, 0.3)
            state[2] = np.random.uniform(0.05, 0.4)
            state[7] = np.random.choice([0.0, 1.0])

            synthetic_states.append(state)

            if np.random.random() < beta or policy_path is None:
                action = np.array([
                    np.random.uniform(-0.01, 0.01),
                    np.random.uniform(-0.01, 0.01),
                    np.random.uniform(-0.01, 0.01),
                    np.random.choice([0.0, 1.0])
                ], dtype=np.float32)
            else:
                action = np.array([
                    np.random.uniform(-0.02, 0.02),
                    np.random.uniform(-0.02, 0.02),
                    np.random.uniform(-0.02, 0.02),
                    np.random.choice([0.0, 1.0])
                ], dtype=np.float32)

            synthetic_actions.append(action)

        synth_data_path = os.path.join(self.dagger_dir, f'synthetic_iter_{iteration}.pkl')
        with open(synth_data_path, 'wb') as f:
            pickle.dump({
                'states': synthetic_states,
                'actions': synthetic_actions
            }, f)

        return synthetic_states, synthetic_actions

    def train_iteration(self, iteration):
        """训练一个迭代"""
        print(f"\n=== DAGGER 迭代 {iteration}/{self.num_iterations} ===")
        synth_states, synth_actions = self.collect_synthetic_data(iteration)

        self.aggregated_data['states'].extend(synth_states)
        self.aggregated_data['actions'].extend(synth_actions)

        print(f"数据聚合: 总样本数 = {len(self.aggregated_data['states'])}")

        from .bc_trainer import BCTrainer

        temp_data_path = os.path.join(self.dagger_dir, f'temp_data_iter_{iteration}.pkl')
        with open(temp_data_path, 'wb') as f:
            pickle.dump({
                'states': np.array(self.aggregated_data['states']),
                'action': np.array(self.aggregated_data['actions'])
            }, f)

        temp_data_dir = os.path.join(self.dagger_dir, f'iter_{iteration}')
        os.makedirs(temp_data_dir, exist_ok=True)
        num_chunks = min(10, len(self.aggregated_data['states']) // 100)
        for i in range(num_chunks):
            chunk_size = len(self.aggregated_data['states']) // num_chunks
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_chunks - 1 else None

            chunk_states = self.aggregated_data['states'][start_idx:end_idx]
            chunk_actions = self.aggregated_data['actions'][start_idx:end_idx]

            chunk_data = {
                'states': np.array(chunk_states),
                'action': np.array(chunk_actions)
            }

            with open(os.path.join(temp_data_dir, f'chunk_{i}.pkl'), 'wb') as f:
                pickle.dump(chunk_data, f)

        trainer = BCTrainer(self.config, temp_data_dir)
        history = trainer.train()

        model_path = os.path.join(self.dagger_dir, f'dagger_iter_{iteration}.pth')
        trainer.save_model(model_path)

        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)

        return trainer.model, history

    def run(self):
        """运行DAGGER算法"""
        print("=" * 50)
        print("开始DAGGER训练")
        print("=" * 50)

        try:
            self.load_expert_data()

            final_model = None

            # DAGGER迭代
            for iteration in range(1, self.num_iterations + 1):
                model, history = self.train_iteration(iteration)
                history_path = os.path.join(self.dagger_dir, f'history_iter_{iteration}.pkl')
                with open(history_path, 'wb') as f:
                    pickle.dump(history, f)

                print(f"迭代 {iteration} 完成")
                final_model = model

            if final_model is None:
                print("警告: 没有训练出最终模型")
                return None

            final_model_path = os.path.join('data', 'models', 'dagger_final_model.pth')
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

            if hasattr(self, 'state_mean') and hasattr(self, 'state_std'):
                state_mean = self.state_mean
                state_std = self.state_std
            else:
                state_mean = np.zeros(8)
                state_std = np.ones(8)

            checkpoint = {
                'model_state_dict': final_model.state_dict(),
                'config': self.config,
                'state_mean': state_mean.tolist() if isinstance(state_mean, np.ndarray) else state_mean,
                'state_std': state_std.tolist() if isinstance(state_std, np.ndarray) else state_std,
                'heteroscedastic': True,
                'dagger_trained': True,
                'dagger_iterations': self.num_iterations,
                'aggregated_data_size': len(self.aggregated_data['states'])
            }

            torch.save(checkpoint, final_model_path)

            if os.path.exists(final_model_path):
                file_size = os.path.getsize(final_model_path) / 1024  # KB
                print(f"DAGGER最终模型已保存到: {final_model_path}")
                print(f"文件大小: {file_size:.1f} KB")
                print(f"聚合数据量: {len(self.aggregated_data['states'])} 个样本")
            else:
                print(f"模型保存失败: {final_model_path}")
                return None

        except Exception as e:
            print(f"DAGGER训练出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        print("\n" + "=" * 50)
        print("DAGGER训练完成!")
        print("=" * 50)

        return final_model