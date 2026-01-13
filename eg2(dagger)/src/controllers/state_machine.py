import time

class PickAndPlaceController:
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env
        self.stage = 0
        self.timer = 0
        
        # 预设的关键点
        self.pick_pos_hover = [0.5, -0.35, 0.3]
        self.pick_pos_down  = [0.5, -0.35, 0.025]
        self.place_pos_hover = [0.8, 0.0, 0.3]
        self.place_pos_down  = [0.8, 0.0, 0.1]
        
        # 固定抓取姿态
        self.orn = [1, 0, 0, 0] 

        
    def reset(self):
        """重置状态机和环境"""
        print(">>> [状态机] 重置演示...")
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

        # 简单的基于时间的有限状态机
        # 1. 移动到物体上方
        if t < 100:
            self.robot.gripper_control(open=True)
            self.robot.move_to(self.pick_pos_hover, self.orn)
            
        # 2. 下降
        elif t < 200:
            self.robot.move_to(self.pick_pos_down, self.orn)
            
        # 3. 抓取
        elif t < 250:
            self.robot.gripper_control(open=False)
            self.robot.move_to(self.pick_pos_down, self.orn)
            
        # 4. 抬起
        elif t < 350:
            self.robot.gripper_control(open=False)
            self.robot.move_to(self.pick_pos_hover, self.orn)
            
        # 5. 移动到目标上方
        elif t < 500:
            self.robot.move_to(self.place_pos_hover, self.orn)
            
        # 6. 下降
        elif t < 600:
            self.robot.move_to(self.place_pos_down, self.orn)
            
        # 7. 松开
        elif t < 650:
            self.robot.gripper_control(open=True)
            self.robot.move_to(self.place_pos_down, self.orn)
            
        # 8. 抬起并复位
        else:
            self.robot.move_to(self.place_pos_hover, self.orn)
            return True
            
        return False