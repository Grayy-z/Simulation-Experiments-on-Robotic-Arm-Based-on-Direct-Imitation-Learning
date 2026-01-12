import time

class PickAndPlaceController:
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env
        self.stage = 0
        self.timer = 0
        
        # === 堆叠任务坐标配置 ===
        # 1. 抓取点 (小物块位置): 根据 env.py 设定在 [0.50, 0.15, 0.015]
        self.pick_pos_hover = [0.50, 0.15, 0.30] 
        self.pick_pos_down  = [0.50, 0.15, 0.015] 
        
        # 2. 放置点 (大物块位置): 根据 env.py 设定在 [0.50, 0.0, 0.025]
        # 放置高度需要注意：大物块顶部高度是 0.05，小物块中心 0.015
        # 理论堆叠中心高度 = 0.05 + 0.015 = 0.065
        # 稍微抬高一点点 (0.07) 防止撞击，让它自然落下
        self.place_pos_hover = [0.50, 0.0, 0.30] 
        self.place_pos_down  = [0.50, 0.0, 0.07]  # 堆叠高度
        
        # 固定抓取姿态 (朝下)
        self.orn = [1, 0, 0, 0] 

    def reset(self):
        """重置状态机和环境"""
        print(">>> [状态机] 重置演示 (堆叠任务)...")
        self.timer = 0
        self.stage = 0
        self.robot.reset()
        self.env.reset_object()

    def update(self):
        """
        每次仿真循环调用一次，根据时间推进状态
        """
        self.timer += 1
        t = self.timer

        # 基于时间的有限状态机 (Time-based State Machine)
        
        # 1. 移动到小物块上方
        if t < 100:
            self.robot.gripper_control(open=True)
            self.robot.move_to(self.pick_pos_hover, self.orn)
            
        # 2. 下降准备抓取
        elif t < 200:
            self.robot.move_to(self.pick_pos_down, self.orn)
            
        # 3. 抓取 (闭合抓手)
        elif t < 250:
            self.robot.gripper_control(open=False)
            self.robot.move_to(self.pick_pos_down, self.orn)
            
        # 4. 抬起小物块
        elif t < 350:
            self.robot.gripper_control(open=False) # 保持闭合
            self.robot.move_to(self.pick_pos_hover, self.orn)
            
        # 5. 移动到大物块(目标)上方
        elif t < 500:
            self.robot.move_to(self.place_pos_hover, self.orn)
            
        # 6. 下降到堆叠位置
        elif t < 600:
            self.robot.move_to(self.place_pos_down, self.orn)
            
        # 7. 松开 (堆叠完成)
        elif t < 650:
            self.robot.gripper_control(open=True)
            self.robot.move_to(self.place_pos_down, self.orn)
            
        # 8. 抬起并复位
        else:
            self.robot.move_to(self.place_pos_hover, self.orn)
            return True # 动作序列完成
            
        return False # 仍在运行