import numpy as np

# 仿真参数
TIME_STEP = 1./240.
GRAVITY = -9.8

# 目标放置区域 (x_min, x_max, y_min, y_max, z_min, z_max)
TARGET_AREA_BOUNDS = [0.7, 0.9, -0.1, 0.1, 0.0, 0.2]

# 物块初始位置
BLOCK_START_POS = [0.5, 0, 0.025]

# 机械臂初始姿态
ARM_RESET_POS = [0, -0.2, 0, -2.0, 0, 2.0, 0.785, 0, 0, 0]