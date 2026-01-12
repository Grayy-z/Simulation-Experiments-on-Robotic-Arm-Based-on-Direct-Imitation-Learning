import pybullet as p
import math

class PandaRobot:
    def __init__(self, base_pos=[0, 0, 0]):
        # 加载 PyBullet 自带的 Franka Panda 机械臂
        # useFixedBase=True 确保基座固定 
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=base_pos, useFixedBase=True)
        
        # 获取关节数量 
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Franka Panda 的末端执行器 link 索引通常是 11
        self.ee_index = 11 
        # 抓手手指的关节索引 (根据 URDF 结构)
        self.finger_indices = [9, 10] 

    def reset(self):
        """重置机器人姿态"""
        rest_poses = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4, 0, 0, 0.04, 0.04]
        for i in range(min(len(rest_poses), self.num_joints)):
            p.resetJointState(self.robot_id, i, rest_poses[i])

    def move_to(self, pos, orn=None):
        """
        移动末端执行器到指定坐标
        Args:
            pos: [x, y, z] 目标位置
            orn: [x, y, z, w] 四元数姿态 (可选)
        """
        # 计算逆运动学 
        if orn:
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_index, pos, orn)
        else:
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_index, pos)

        # 应用控制信号 - 使用 setJointMotorControlArray 提高效率
        # 注意：Panda 前7个是机械臂关节，后面是抓手
        for i in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500  # 提供足够的力
            )

    def gripper_control(self, open=True):
        """
        控制抓手开合
        Args:
            open: True 为张开，False 为闭合
        """
        target_val = 0.04 if open else 0.0
        force = 20 if open else 50 # 抓取时给大一点力
        
        for i in self.finger_indices:
            p.setJointMotorControl2(
                self.robot_id, 
                i, 
                p.POSITION_CONTROL, 
                targetPosition=target_val, 
                force=force
            )
            
    def get_gripper_width(self):
        """
        获取当前抓手的张开宽度 [新增方法]
        Returns:
            float: 两个指关节位置之和 (单位: 米)
        """
        # p.getJointState 返回 (pos, vel, reaction_forces, applied_torque)
        finger1_state = p.getJointState(self.robot_id, self.finger_indices[0])
        finger2_state = p.getJointState(self.robot_id, self.finger_indices[1])
        
        # 两个指关节都是移动关节 (0 ~ 0.04)，宽度是两者之和
        # 完全张开时约为 0.08
        return finger1_state[0] + finger2_state[0]