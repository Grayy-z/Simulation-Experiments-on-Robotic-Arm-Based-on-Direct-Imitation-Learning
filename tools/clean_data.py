import os
import json
import numpy as np
import pickle
from tqdm import tqdm

# === 配置区域 ===
SOURCE_DIR = 'data/trajectories'
TARGET_DIR = 'data/cleaned_trajectories'

# 核心修改：不再使用时间间隔，而是使用距离间隔
# 只有当机械臂累计移动了 0.003m 或者夹爪状态改变时，才记录一帧
MIN_DISTANCE_THRESHOLD = 0.003

def load_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取错误 {path}: {e}")
        return None

def process_raw_list_data(raw_data):
    """解析原始数据列表"""
    if not isinstance(raw_data, list): return None, None
    
    # 提取所有帧的 pos, orn, gripper
    all_pos = []
    all_orn = []
    all_grip = []
    
    for frame in raw_data:
        all_pos.append(frame['pos'])
        all_orn.append(frame['orn'])
        all_grip.append(1.0 if frame['gripper_open'] else 0.0)
        
    return np.array(all_pos), np.array(all_orn), np.array(all_grip)

def clean_trajectory(traj_data, filename):
    # 1. 解析原始数据
    raw_pos, raw_orn, raw_grip = process_raw_list_data(traj_data)
    if raw_pos is None or len(raw_pos) < 2: return None

    cleaned_states = []
    cleaned_actions = []

    # === 核心算法：基于距离的重采样 ===
    
    # 初始化
    last_saved_pos = raw_pos[0]
    last_saved_grip = raw_grip[0]
    
    # 暂存当前累积的状态
    # 我们总是保存"上一个关键帧"作为 State，"当前关键帧与上一个的差"作为 Action
    
    current_state_buffer = list(raw_pos[0]) + list(raw_orn[0]) + [raw_grip[0]]
    
    for i in range(1, len(raw_pos)):
        curr_pos = raw_pos[i]
        curr_grip = raw_grip[i]
        
        # 计算与上一次保存点的距离
        dist = np.linalg.norm(curr_pos - last_saved_pos)
        
        # 判断夹爪是否变化 (这是一个极其重要的关键事件)
        grip_changed = abs(curr_grip - last_saved_grip) > 0.5
        
        # === 判定条件 ===
        # 1. 移动距离超过阈值 (过滤掉微小抖动和静止)
        # 2. 或者 夹爪发生了开合 (必须记录)
        # 3. 或者 是最后一帧 (保证完整性)
        if dist >= MIN_DISTANCE_THRESHOLD or grip_changed or i == len(raw_pos) - 1:
            
            # --- 构造样本对 ---
            # Input State: 上一个保存点的绝对状态
            state = np.array(current_state_buffer, dtype=np.float32)
            
            # Output Action: 当前点 - 上一个保存点 (这就是一步"大"动作)
            # Action = [dx, dy, dz, gripper_state]
            delta_pos = curr_pos - last_saved_pos
            action = list(delta_pos) + [curr_grip]
            action = np.array(action, dtype=np.float32)
            
            cleaned_states.append(state)
            cleaned_actions.append(action)
            
            # --- 更新状态 ---
            last_saved_pos = curr_pos
            last_saved_grip = curr_grip
            # 更新 State buffer 为当前这一帧，作为下一次的起点
            current_state_buffer = list(curr_pos) + list(raw_orn[i]) + [curr_grip]

    # 再次检查数据有效性
    if len(cleaned_states) < 5:
        print(f"[{filename}] 警告: 有效动作太少，已丢弃")
        return None

    return {
        'states': np.array(cleaned_states, dtype=np.float32),
        'action': np.array(cleaned_actions, dtype=np.float32)
    }

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"目录不存在: {SOURCE_DIR}")
        return
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]
    print(f"发现 {len(files)} 个JSON文件，开始基于距离的重采样...")

    count = 0
    total_frames_before = 0
    total_frames_after = 0

    for fname in tqdm(files):
        path = os.path.join(SOURCE_DIR, fname)
        data = load_json_file(path)
        if data is None: continue
        
        total_frames_before += len(data)
        
        new_data = clean_trajectory(data, fname)
        
        if new_data:
            total_frames_after += len(new_data['states'])
            save_name = fname.replace('.json', '.pkl')
            save_path = os.path.join(TARGET_DIR, save_name)
            with open(save_path, 'wb') as f:
                pickle.dump(new_data, f)
            count += 1

    print(f"\n处理完成: {count} / {len(files)}")
    print(f"原始帧数: {total_frames_before}")
    print(f"清洗后帧数: {total_frames_after}")
    print(f"压缩比: {total_frames_after/total_frames_before*100:.2f}% (保留了最有意义的关键动作)")
    print("现在请重新运行 train_bc.py！")

if __name__ == '__main__':
    main()