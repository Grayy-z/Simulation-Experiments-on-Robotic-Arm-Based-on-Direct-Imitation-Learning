import pybullet as p
import pybullet_data
import time
from config.settings import TIME_STEP, GRAVITY
from src.utils.tools import draw_target_area

class SimulationEnv:
    def __init__(self, gui=True):
        connection_mode = p.GUI if gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(TIME_STEP)

        self.big_block_id = None
        self.small_block_id = None
        self.plane_id = None
        
        # 物块初始位置
        self.FIXED_BIG_BLOCK_POS = [0.50, 0.0, 0.033] 
        self.FIXED_SMALL_BLOCK_POS = [0.60, 0.0, 0.025]

    def setup_scene(self):
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.01, rollingFriction=0.01)
        # 1. 创建大方块 (红色, 半边长 0.025)
        big_half_extents = [0.033, 0.033, 0.033]
        big_col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=big_half_extents)
        big_vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=big_half_extents, rgbaColor=[1, 0, 0, 1])
        
        self.big_block_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=big_col_id,
            baseVisualShapeIndex=big_vis_id,
            basePosition=self.FIXED_BIG_BLOCK_POS
        )

        # 2. 创建小方块 (绿色, 半边长 0.015)
        small_half_extents = [0.025, 0.025, 0.025]
        small_col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=small_half_extents)
        small_vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=small_half_extents, rgbaColor=[0, 1, 0, 1])
        
        self.small_block_id = p.createMultiBody(
            baseMass=0.05, # 质量较轻
            baseCollisionShapeIndex=small_col_id,
            baseVisualShapeIndex=small_vis_id,
            basePosition=self.FIXED_SMALL_BLOCK_POS
        )
        
        draw_target_area()

    def reset_object(self):
        """重置所有物块位置"""
        if self.big_block_id is not None:
            p.resetBasePositionAndOrientation(self.big_block_id, self.FIXED_BIG_BLOCK_POS, [0, 0, 0, 1])
        
        if self.small_block_id is not None:
            p.resetBasePositionAndOrientation(self.small_block_id, self.FIXED_SMALL_BLOCK_POS, [0, 0, 0, 1])
            
        return self.FIXED_BIG_BLOCK_POS, self.FIXED_SMALL_BLOCK_POS

    def step(self):
        p.stepSimulation()
        time.sleep(TIME_STEP)

    def disconnect(self):
        p.disconnect()
        
    def get_block_positions(self):
        """返回两个物块的位置信息"""
        big_pos, big_orn = ([0,0,0], [0,0,0,1])
        small_pos, small_orn = ([0,0,0], [0,0,0,1])

        if self.big_block_id is not None:
            big_pos, big_orn = p.getBasePositionAndOrientation(self.big_block_id)
        if self.small_block_id is not None:
            small_pos, small_orn = p.getBasePositionAndOrientation(self.small_block_id)
            
        return (big_pos, big_orn), (small_pos, small_orn)