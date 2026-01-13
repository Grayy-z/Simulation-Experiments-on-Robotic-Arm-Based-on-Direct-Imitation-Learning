import pybullet as p
import pybullet_data
import time
from config.settings import TIME_STEP, GRAVITY, BLOCK_START_POS
from src.utils.tools import draw_target_area

class SimulationEnv:
    def __init__(self, gui=True):
        connection_mode = p.GUI if gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(TIME_STEP)

        self.block_id = None
        self.plane_id = None
        self.table_id = None
        
        # === 物块位置 ===
        self.FIXED_BLOCK_POS = [0.50, -0.35, 0.025]

    def setup_scene(self):
        self.plane_id = p.loadURDF("plane.urdf")

        
        # 创建方块
        half_extents = [0.025, 0.025, 0.025]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1])
        
        self.block_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=self.FIXED_BLOCK_POS
        )
        
        draw_target_area()

    def reset_object(self):
        if self.block_id is not None:
            p.resetBasePositionAndOrientation(self.block_id, self.FIXED_BLOCK_POS, [0, 0, 0, 1])
            return self.FIXED_BLOCK_POS
        return None
        

    def step(self):
        p.stepSimulation()
        time.sleep(TIME_STEP)

    def disconnect(self):
        p.disconnect()
        
    def get_block_position(self):
        if self.block_id is not None:
            return p.getBasePositionAndOrientation(self.block_id)
        return [0,0,0], [0,0,0,1]