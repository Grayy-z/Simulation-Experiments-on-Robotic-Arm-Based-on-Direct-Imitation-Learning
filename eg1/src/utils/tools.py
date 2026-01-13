import pybullet as p
from config.settings import TARGET_AREA_BOUNDS

def successful(object_id):
    """
    检查物体是否位于目标区域内，并且已放置在地面上。
    
    Args:
        object_id: 物块在 PyBullet 中的 ID
    
    Returns:
        bool: 只有当物体在 XY 框内 且 高度 Z 足够低时 返回 True
    """
    pos, _ = p.getBasePositionAndOrientation(object_id)
    x, y, z = pos

    x_min, x_max, y_min, y_max, z_min, z_max = TARGET_AREA_BOUNDS

    ON_GROUND_THRESHOLD = 0.05 

    xy_success = (x_min <= x <= x_max) and (y_min <= y <= y_max)
    z_success = z <= ON_GROUND_THRESHOLD

    return xy_success and z_success

def draw_target_area():
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
