import pybullet as p
import math
from config.settings import TARGET_AREA_BOUNDS

def successful(big_block_id, small_block_id, is_gripper_released):
    """
    检查任务是否成功。成功条件：
    1. 大物块在目标区域内且在地面上
    2. 小物块在目标区域内
    3. 小物块叠在大物块上方 (Z轴高度差合适，且XY平面距离接近)
    4. 抓手状态为放开 (is_gripper_released 为 True)
    
    Args:
        big_block_id: 大物块 ID
        small_block_id: 小物块 ID
        is_gripper_released (bool): 抓手是否处于张开状态 (需要主循环传入，例如判断 width > threshold)
    Returns:
        bool: 是否成功
    """
    # 0. 如果抓手没放开，直接返回 False
    if not is_gripper_released:
        return False

    # 获取位置
    pos_big, _ = p.getBasePositionAndOrientation(big_block_id)
    pos_small, _ = p.getBasePositionAndOrientation(small_block_id)
    
    x_min, x_max, y_min, y_max, z_min, z_max = TARGET_AREA_BOUNDS
    
    # === 条件 1: 大物块在目标框内且贴地 ===
    big_xy_ok = (x_min <= pos_big[0] <= x_max) and (y_min <= pos_big[1] <= y_max)
    big_z_ok = pos_big[2] <= 0.04 # 允许一点点误差，大物块半高是0.025
    
    if not (big_xy_ok and big_z_ok):
        return False

    # === 条件 2: 小物块在目标框内 ===
    small_xy_in_bounds = (x_min <= pos_small[0] <= x_max) and (y_min <= pos_small[1] <= y_max)
    
    if not small_xy_in_bounds:
        return False

    # === 条件 3: 堆叠判断 (Stacking Logic) ===
    # 3.1 垂直高度判断: 小物块Z 应该大于 大物块Z
    # 大物块高0.05(中心0.025)，小物块高0.03(中心0.015)。堆叠后小物块中心理论高度约为 0.05 + 0.015 = 0.065
    # 我们设定小物块中心必须高于 0.05 (确保它在上面，而不是旁边)
    is_stacked_z = pos_small[2] > 0.05

    # 3.2 水平对齐判断: 两个物块的中心在XY平面的距离应该很小
    xy_distance = math.sqrt((pos_big[0] - pos_small[0])**2 + (pos_big[1] - pos_small[1])**2)
    # 允许有一定的错位，例如 3cm 以内
    is_stacked_xy = xy_distance < 0.03

    return is_stacked_z and is_stacked_xy

def draw_target_area():
    """在 GUI 中画出目标区域的线框"""
    x_min, x_max, y_min, y_max, z_min, z_max = TARGET_AREA_BOUNDS
    
    corners = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ]
    
    p.addUserDebugLine(corners[0], corners[1], [1, 0, 0], 2)
    p.addUserDebugLine(corners[1], corners[2], [1, 0, 0], 2)
    p.addUserDebugLine(corners[2], corners[3], [1, 0, 0], 2)
    p.addUserDebugLine(corners[3], corners[0], [1, 0, 0], 2)
