import pybullet as p
import math

class PandaRobot:
    def __init__(self, base_pos=[0, 0, 0]):
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=base_pos, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.ee_index = 11 
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
            orn: [x, y, z, w] 四元数姿态
        """
        # 计算逆运动学 
        if orn:
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_index, pos, orn)
        else:
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_index, pos)

        for i in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500
            )

    def gripper_control(self, open=True):
        """
        控制抓手开合
        Args:
            open: True 为张开，False 为闭合
        """
        target_val = 0.04 if open else 0.0
        force = 20 if open else 50
        
        for i in self.finger_indices:
            p.setJointMotorControl2(
                self.robot_id, 
                i, 
                p.POSITION_CONTROL, 
                targetPosition=target_val, 
                force=force
            )