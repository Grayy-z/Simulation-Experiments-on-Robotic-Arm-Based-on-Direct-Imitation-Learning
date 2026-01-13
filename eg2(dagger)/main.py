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
    print("\n=== 机器人抓取仿真系统 ===")
    print("1. 自动抓取演示 (State Machine)")
    print("2. 录制模式 (Keyboard)")
    print("3. 推理模式 (BC Model)")
    print("4. DAGGER推理模式")
    
    mode = input("请输入模式编号 (1/2/3/4): ").strip()

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
            controller = BCController(robot, sim, model_path)
            print(f"\n加载模型: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"错误: {e}")
            return

    elif mode == '4':  # DAGGER控制器
        default_model = os.path.join('data', 'models', 'best_bc_model.pth')
        model_path = input(f"DAGGER模型路径 [{default_model}]: ").strip() or default_model

        if not os.path.isabs(model_path) and '/' not in model_path and '\\' not in model_path:
            model_path = os.path.join('data', 'models', model_path)

        try:
            from src.controllers.dagger_controller import DAGGERController
            controller = DAGGERController(robot, sim, model_path)
            print(f"\n加载DAGGER模型: {os.path.basename(model_path)}")
            print("高不确定性时会自动采取保守策略")
        except Exception as e:
            print(f"错误: {e}")
            return

    else:
        controller = PickAndPlaceController(robot, sim)

    print("系统运行中... (Ctrl+C 退出)")
    
    task_success = False

    try:
        while True:
  
            done = controller.update()

            sim.step()

            # 检测是否成功
            if successful(sim.block_id):
                if not task_success:
                    print("\n>>> 成功放入目标区域！")
                    if mode == '2':
                        print(">>> 你现在可以按 [空格] 松开夹爪，或按 [T] 重置。")
                    task_success = True
            
            else:
                task_success = False

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