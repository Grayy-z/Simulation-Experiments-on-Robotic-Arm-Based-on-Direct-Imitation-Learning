import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_trajectory_stats(data_dir):
    """
    分析清洗后的轨迹数据
    """
    traj_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    if not traj_files:
        print(f"在 {data_dir} 中未找到 .pkl 文件")
        return {
            'num_trajectories': 0,
            'total_points': 0,
            'avg_traj_length': 0,
            'pos_change_stats': {'min':0, 'max':0, 'mean':0, 'std':0}
        }

    all_delta_pos = []
    all_gripper_changes = []
    traj_lengths = []

    for file in traj_files:
        filepath = os.path.join(data_dir, file)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # 提取数据
        states = data['states']
        actions = data['action']
        
        traj_lengths.append(len(states))

        # 计算位置变化
        delta_pos = actions[:, :3]
        all_delta_pos.extend(delta_pos.flatten())

        # 计算抓手状态变化
        if len(states) > 1:
            gripper_states = states[:, 7]
            gripper_changes = np.diff(gripper_states)
            changes = gripper_changes[gripper_changes != 0]
            all_gripper_changes.extend(changes.tolist())

    all_delta_pos = np.array(all_delta_pos) if all_delta_pos else np.array([0])
    all_gripper_changes = np.array(all_gripper_changes) if all_gripper_changes else np.array([0])

    print(f"轨迹数量: {len(traj_files)}")
    print(f"总数据点: {sum(traj_lengths)}")
    print(f"平均轨迹长度: {np.mean(traj_lengths):.2f}")
    print(f"位置变化范围: [{np.min(all_delta_pos):.6f}, {np.max(all_delta_pos):.6f}]")
    print(f"位置变化均值: {np.mean(all_delta_pos):.6f}")
    
    gripper_change_count = np.sum(np.abs(all_gripper_changes) > 0.5)
    print(f"抓手变化次数: {gripper_change_count}")

    try:
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.hist(all_delta_pos, bins=50, alpha=0.7)
        plt.title('Action (Delta Pos) Distribution')
        plt.xlabel('Meters')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.hist(traj_lengths, bins=20, alpha=0.7)
        plt.title('Trajectory Lengths')
        plt.xlabel('Steps')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.hist(all_gripper_changes, bins=10, alpha=0.7)
        plt.title('Gripper Changes')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('trajectory_analysis.png', dpi=100)
        print("分析图表已保存至 trajectory_analysis.png")
    except Exception as e:
        print(f"绘图失败: {e}")

    return {
        'num_trajectories': len(traj_files),
        'total_points': sum(traj_lengths),
        'avg_traj_length': np.mean(traj_lengths),
        'pos_change_stats': {
            'min': float(np.min(all_delta_pos)),
            'max': float(np.max(all_delta_pos)),
            'mean': float(np.mean(all_delta_pos)),
            'std': float(np.std(all_delta_pos))
        }
    }