# check_trained_model.py
import torch
import numpy as np
import os
import sys
import json


def load_and_analyze_model(model_path):
    """加载和分析训练好的模型"""
    print(f"正在检查模型文件: {model_path}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f" 模型文件不存在: {model_path}")
        return None

    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        print("模型文件加载成功")
        print(f"检查点包含的键: {list(checkpoint.keys())}")

        # 配置信息
        print("\n" + "=" * 60)
        print("1. 模型配置信息:")
        print("=" * 60)
        if 'config' in checkpoint:
            config = checkpoint['config']
            for key, value in config.items():
                print(f"  {key}: {value}")

            expected_config = {
                'input_dim': 8,
                'output_dim': 4,
                'seq_length': 10
            }

            print(f"\n关键配置检查:")
            for key, expected in expected_config.items():
                actual = config.get(key)
                if actual is not None:
                    if actual == expected:
                        print(f"{key}: {actual} (符合期望)")
                    else:
                        print(f"{key}: {actual} (期望: {expected})")
                else:
                    print(f" {key}: 缺失")
        else:
            print(" 配置信息缺失")

        print("\n" + "=" * 60)
        print("2. 数据标准化参数:")
        print("=" * 60)

        if 'state_mean' in checkpoint and 'state_std' in checkpoint:
            state_mean = checkpoint['state_mean']
            state_std = checkpoint['state_std']

            if isinstance(state_mean, list):
                state_mean = np.array(state_mean)
            if isinstance(state_std, list):
                state_std = np.array(state_std)

            print(f"state_mean 形状: {state_mean.shape}")
            print(f"state_std 形状: {state_std.shape}")

            # 检查零标准差
            zero_std_indices = np.where(state_std == 0)[0]
            if len(zero_std_indices) > 0:
                print(f"发现 {len(zero_std_indices)} 个零标准差的位置: {zero_std_indices}")
                print(f"  这将导致标准化时除以零！")

            # 显示各维度的统计信息
            print(f"\n各维度标准化参数:")
            print(f"{'维度':<6} {'均值':<15} {'标准差':<15}")
            for i in range(len(state_mean)):
                print(f"{i:<6} {state_mean[i]:<15.6f} {state_std[i]:<15.6f}")
        else:
            print("标准化参数缺失")

        # 检查训练历史
        print("\n" + "=" * 60)
        print("3. 训练历史:")
        print("=" * 60)

        if 'history' in checkpoint:
            history = checkpoint['history']
            if 'train_loss' in history and 'val_loss' in history:
                train_losses = history['train_loss']
                val_losses = history['val_loss']

                print(f"训练轮次: {len(train_losses)}")
                print(f"初始训练损失: {train_losses[0]:.6f}")
                print(f"最终训练损失: {train_losses[-1]:.6f}")
                print(f"训练损失改善: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")

                if len(val_losses) > 0:
                    print(f"\n初始验证损失: {val_losses[0]:.6f}")
                    print(f"最终验证损失: {val_losses[-1]:.6f}")
                    print(f"最佳验证损失: {min(val_losses):.6f}")

                    # 检查过拟合
                    if train_losses[-1] < val_losses[-1]:
                        print(
                            f" 可能存在过拟合: 训练损失({train_losses[-1]:.6f}) < 验证损失({val_losses[-1]:.6f})")
            else:
                print(" 训练历史数据不完整")
        else:
            print(" 训练历史缺失")

        # 检查模型权重
        print("\n" + "=" * 60)
        print("4. 模型权重信息:")
        print("=" * 60)

        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']

            print(f"模型参数量: {len(model_state.keys())}")

            # 检查关键层
            total_params = 0
            trainable_params = 0

            for key, tensor in model_state.items():
                params = tensor.numel()
                total_params += params

                print(f"  {key}: {list(tensor.shape)} | 参数: {params:,}")
                if 'lstm.weight_ih_l0' in key:
                    input_dim = tensor.shape[1]
                    hidden_dim = tensor.shape[0] // 4 
                    print(f"  LSTM输入维度检测: {input_dim}")
                    print(f"  LSTM隐藏层维度检测: {hidden_dim}")

            print(f"\n总参数量: {total_params:,}")

            has_nan = False
            has_inf = False

            for key, tensor in model_state.items():
                if torch.isnan(tensor).any():
                    has_nan = True
                    print(f"{key} 包含NaN值")
                if torch.isinf(tensor).any():
                    has_inf = True
                    print(f"{key} 包含Inf值")

            if not has_nan and not has_inf:
                print("权重中没有NaN或Inf值")

        print("\n" + "=" * 60)
        print("5. 模型推理测试:")
        print("=" * 60)


        class SimpleBCNetwork(torch.nn.Module):
            def __init__(self, input_dim=8, output_dim=4, hidden_size=128, num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc_layers = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size * 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, output_dim)
                )

            def forward(self, x):
                lstm_out, (hidden, cell) = self.lstm(x)
                last_output = lstm_out[:, -1, :]
                action = self.fc_layers(last_output)
                return action

        if 'config' in checkpoint:
            config = checkpoint['config']

            # 创建模型并加载权重
            model = SimpleBCNetwork(
                input_dim=config.get('input_dim', 8),
                output_dim=config.get('output_dim', 4),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # 创建测试输入
            batch_size = 1
            seq_length = config.get('seq_length', 10)
            input_dim = config.get('input_dim', 8)

            # 随机输入
            test_input = torch.randn(batch_size, seq_length, input_dim)

            # 标准化输入
            if 'state_mean' in checkpoint and 'state_std' in checkpoint:
                state_mean = checkpoint['state_mean']
                state_std = checkpoint['state_std']

                if isinstance(state_mean, list):
                    state_mean = torch.tensor(state_mean)
                if isinstance(state_std, list):
                    state_std = torch.tensor(state_std)

                # 应用标准化
                test_input = (test_input - state_mean) / state_std

            # 前向传播
            with torch.no_grad():
                output = model(test_input)

            print(f"测试输入形状: {test_input.shape}")
            print(f"模型输出形状: {output.shape}")
            print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
            print(f"输出平均值: {output.mean():.6f}")
            print(f"输出标准差: {output.std():.6f}")

            # 分析输出
            print(f"\n输出维度分析:")
            print(f"  delta_x: {output[0, 0]:.6f}")
            print(f"  delta_y: {output[0, 1]:.6f}")
            print(f"  delta_z: {output[0, 2]:.6f}")
            print(f"  gripper_action: {output[0, 3]:.6f}")

            # 检查gripper_action范围合理性
            gripper_value = output[0, 3].item()
            if gripper_value > 0.5:
                print(f"Gripper动作: {gripper_value:.4f} (倾向于张开)")
            else:
                print(f"Gripper动作: {gripper_value:.4f} (倾向于闭合)")

        print("\n" + "=" * 60)
        print("检查完成!")
        print("=" * 60)

        return checkpoint

    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(model_paths):
    """比较多个模型文件"""
    print("比较多个模型文件:")
    print("=" * 60)

    results = {}

    for model_path in model_paths:
        print(f"\n分析: {model_path}")

        checkpoint = load_and_analyze_model(model_path)
        if checkpoint and 'config' in checkpoint:
            config = checkpoint['config']
            key = os.path.basename(model_path)
            results[key] = {
                'input_dim': config.get('input_dim', 'N/A'),
                'output_dim': config.get('output_dim', 'N/A'),
                'seq_length': config.get('seq_length', 'N/A'),
                'val_loss': min(checkpoint.get('history', {}).get('val_loss', [float('inf')]))
            }

    print("\n" + "=" * 60)
    print("模型比较结果:")
    print("=" * 60)
    print(f"{'模型文件':<30} {'输入维度':<10} {'输出维度':<10} {'序列长度':<10} {'最佳验证损失':<15}")

    for model_name, info in results.items():
        print(
            f"{model_name:<30} {info['input_dim']:<10} {info['output_dim']:<10} {info['seq_length']:<10} {info['val_loss']:<15.6f}")

    return results


def main():
    """主函数"""
    print("BC模型分析工具")
    print("=" * 60)


    default_models = [
        os.path.join('data', 'models', 'best_bc_model.pth'),
        os.path.join('data', 'models', 'final_bc_model.pth'),
        os.path.join('models', 'best_bc_model.pth'),
    ]

    available_models = []
    for model_path in default_models:
        if os.path.exists(model_path):
            available_models.append(model_path)

    if not available_models:
        print("没有找到模型文件")
        print("请先运行 train_bc.py 训练模型")
        return

    print(f"找到 {len(available_models)} 个模型文件:")
    for i, model_path in enumerate(available_models, 1):
        print(f"  {i}. {model_path}")

    print("\n选择操作:")
    print("  1. 分析所有模型")
    print("  2. 分析指定模型")
    print("  3. 比较所有模型")

    choice = input("请输入选择 (1-3): ").strip()

    if choice == '1':
        for model_path in available_models:
            load_and_analyze_model(model_path)
            print("\n")
    elif choice == '2':
        print("\n可用的模型:")
        for i, model_path in enumerate(available_models, 1):
            print(f"  {i}. {model_path}")

        model_choice = input("请选择模型编号: ").strip()
        try:
            idx = int(model_choice) - 1
            if 0 <= idx < len(available_models):
                load_and_analyze_model(available_models[idx])
            else:
                print("无效的选择")
        except:
            print("无效的输入")
    elif choice == '3':
        compare_models(available_models)
    else:
        print("无效的选择")


if __name__ == "__main__":
    main()