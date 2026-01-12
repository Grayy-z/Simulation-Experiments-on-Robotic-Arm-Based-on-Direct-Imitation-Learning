# train_bc.py
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(__file__))

from config.bc_config import BC_CONFIG
from src.utils.bc_trainer import BCTrainer
import matplotlib.pyplot as plt

def plot_training_history(history):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))

    # 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # 验证损失
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mse'], label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)

    # 保存路径
    save_dir = BC_CONFIG.get('models_dir', 'data/models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_history.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练历史图表已保存到: {save_path}")
    plt.show()

def main():
    """主训练函数"""
    print("=" * 50)
    print("开始训练行为克隆模型 (Loading .pkl data)")
    print("=" * 50)

    # 轨迹数据目录 (使用清洗后的数据)
    data_dir = os.path.join('data', 'cleaned_trajectories')

    # 模型保存目录
    models_dir = BC_CONFIG['models_dir']
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        print(f"轨迹数据目录不存在: {data_dir}")
        print("请先运行 clean_data.py 清洗数据")
        return

    print("\n=== 数据分析 ===")

    # 先分析数据
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
        from test.analyze_trajectory_stats import analyze_trajectory_stats

        print("正在分析轨迹数据...")
        stats = analyze_trajectory_stats(data_dir)

        # 只有在分析成功且有数据时才打印
        if stats and stats['num_trajectories'] > 0:
            print(f"\n分析结果:")
            print(f"  轨迹文件数量: {stats['num_trajectories']}")
            print(f"  总数据点数: {stats['total_points']}")
            print(f"  平均轨迹长度: {stats['avg_traj_length']:.2f}")
            print(f"  位置变化统计:")
            print(f"    最小值: {stats['pos_change_stats']['min']:.6f}")
            print(f"    最大值: {stats['pos_change_stats']['max']:.6f}")
            print(f"    均值: {stats['pos_change_stats']['mean']:.6f}")
            print(f"    标准差: {stats['pos_change_stats']['std']:.6f}")
        else:
            print("警告: 分析结果为空，可能未找到有效数据。")

    except Exception as e:
        print(f"数据分析时出错 (不影响训练): {e}")
        # import traceback
        # traceback.print_exc()

    # 询问是否继续训练
    proceed = input("\n是否继续训练? (输入 y 继续，其他键取消): ").lower().strip()
    if proceed != 'y':
        print("训练取消")
        return

    # === 修改点：统计 .pkl 文件而不是 .json ===
    traj_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"\n找到 {len(traj_files)} 个清洗后的轨迹文件 (.pkl)")

    if len(traj_files) == 0:
        print("没有找到 .pkl 轨迹数据文件")
        print("请检查 data/cleaned_trajectories 目录是否为空，或重新运行 clean_data.py")
        return

    # 创建训练器并开始训练
    try:
        trainer = BCTrainer(BC_CONFIG, data_dir)
        history = trainer.train()

        # 绘制训练曲线
        plot_training_history(history)

        print("=" * 50)
        print("训练完成！")
        print(f"\n模型已保存到: {models_dir}/")
        print(f"  最佳模型: {BC_CONFIG['best_model_name']}")
        print(f"  最终模型: {BC_CONFIG['final_model_name']}")
        print("=" * 50)

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()