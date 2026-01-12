import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.sim.env import SimulationEnv
from src.robot.panda_arm import PandaRobot
from src.controllers.state_machine import PickAndPlaceController
from src.controllers.keyboard_ctrl import KeyboardController
from src.controllers.bc_controller import BCController
from src.utils.tools import successful


def main():
    print("\n=== 机器人堆叠仿真系统 (双物块) ===")
    print("1. 自动抓取演示 (State Machine)")
    print("2. 录制模式 (Keyboard)")
    print("3. 推理模式 (BC Model)")
    
    mode = input("请输入模式编号 (1/2/3): ").strip()

    sim = SimulationEnv(gui=True)
    sim.setup_scene()

    robot = PandaRobot(base_pos=[0, 0, 0])
    robot.reset()

    if mode == '2':

        controller = KeyboardController(robot, sim)
        print("\n=== 录制指南 ===")
        print(" [R] 键: 开始/停止录制")
        print(" [T] 键: 重置环境 (随时可用)") 
        print(" [空格]: 切换夹爪")
        print("================\n")

    elif mode == '3':
        default_model = os.path.join('data', 'models', 'best_bc_model.pth')
        model_path = input(f"模型路径 [{default_model}]: ").strip() or default_model
        
        if not os.path.isabs(model_path) and '/' not in model_path and '\\' not in model_path:
            model_path = os.path.join('data', 'models', model_path)

        try:
            # 注意：BCController 需要修改以接收两个物块的状态作为输入
            controller = BCController(robot, sim, model_path)
            print(f"\n加载模型: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"错误: {e}")
            return

    else:
        # 注意：状态机逻辑如果是硬编码的，可能无法自动处理堆叠任务
        controller = PickAndPlaceController(robot, sim)

    print("系统运行中... (Ctrl+C 退出)")
    
    task_success = False

    try:
        while True:
            # 获取动作并更新
            done = controller.update()
            sim.step()

            current_width = robot.get_gripper_width() 
            is_released = current_width > 0.07 
            
            if successful(sim.big_block_id, sim.small_block_id, is_released):
                if not task_success:
                    print("\n>>> 任务成功！(物块已堆叠且抓手已松开)")
                    if mode == '2':
                        print(">>> 演示完成，按 [T] 重置环境。")
                    task_success = True
            else:
                task_success = False
            
            # === 修改结束 ===

            # 状态机模式下的自动演示重置
            if mode == '1' and done:
                print("演示结束，自动重置...")
                time.sleep(2)

                if hasattr(controller, 'reset'):
                    controller.reset()

    except KeyboardInterrupt:
        print("\n程序中断")
    finally:
        sim.disconnect()

if __name__ == "__main__":
    main()