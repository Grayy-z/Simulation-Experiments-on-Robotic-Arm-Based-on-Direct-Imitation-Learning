import pybullet as p
from src.utils.recorder import TrajectoryRecorder

class KeyboardController:
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env
        self.recorder = TrajectoryRecorder()
        
        self.init_pos = [0.5, 0, 0.3]
        self.init_orn = [1, 0, 0, 0]
        
        self.reset_internal_state()
        
        self.step_size = 0.001

    def reset_internal_state(self):
        """重置控制器的内部变量"""
        self.current_pos = list(self.init_pos)
        self.current_orn = list(self.init_orn)
        self.gripper_open = True 

    def reset(self):
        """执行全面重置：环境、机器人、控制器状态"""
        print(">>> [控制器] 执行重置操作...")
        
        self.reset_internal_state()
        
        self.robot.reset()
        
        self.robot.gripper_control(open=True)
        
        self.env.reset_object()

    def update(self):
        keys = p.getKeyboardEvents()
        
        if ord('t') in keys and (keys[ord('t')] & p.KEY_WAS_TRIGGERED):
            self.reset()
            return False

        if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN):
            self.current_pos[0] += self.step_size
        if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN):
            self.current_pos[0] -= self.step_size
            
        if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN):
            self.current_pos[1] -= self.step_size
        if p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN):
            self.current_pos[1] += self.step_size

        if ord('q') in keys and (keys[ord('q')] & p.KEY_IS_DOWN):
            self.current_pos[2] += self.step_size
        if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN):
            self.current_pos[2] -= self.step_size

        if ord(' ') in keys and (keys[ord(' ')] & p.KEY_WAS_TRIGGERED):
            self.gripper_open = not self.gripper_open
            print(f"抓手状态: {'张开' if self.gripper_open else '闭合'}")

        if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
            if self.recorder.is_recording:
                self.recorder.stop()
            else:
                self.recorder.start()

        if self.current_pos[2] < 0.0: self.current_pos[2] = 0.0

        self.robot.move_to(self.current_pos, self.current_orn)
        self.robot.gripper_control(open=self.gripper_open)

        self.recorder.record_step(
            list(self.current_pos), 
            list(self.current_orn), 
            self.gripper_open
        )
        
        return False