import os
import torch

MODELS_DIR = os.path.join('data', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

BC_CONFIG = {
    # 训练参数
    'batch_size': 32,
    'learning_rate': 0.001,  
    'num_epochs': 100,        
    'train_split': 0.8,
    'patience': 15,

    # 模型结构
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,

    # 输入输出维度
    'input_dim': 8,
    'output_dim': 8,

    # 异方差模型
    'heteroscedastic': True,
    'variance_regularization': 0.01,  # 新增：方差正则化系数

    # Loss权重（调整以平衡MSE和正则化）
    'loss_weights': {
        'pos': 1.0,
        'pos_z': 10.0,
        'gripper': 100.0,
        'variance_reg': 0.01  # 新增：方差正则化权重
    },

    # 新增时序感知配置
    'temporal_aware': True,  # 启用时序感知
    'time_encoding_dim': 4,  # 时间编码维度
    'direction_weight': 0,  # 方向感知损失权重
    'variance_peak_weight': 0,  # 鼓励在转折点提高方差

    # 数据参数
    'seq_length': 10,       
    'normalize': True,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'models_dir': MODELS_DIR,
    'best_model_name': 'best_bc_model.pth',
    'final_model_name': 'final_bc_model.pth'
}