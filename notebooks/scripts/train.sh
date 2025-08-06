#!/bin/bash
#SBATCH --job-name=toy_exp          # 作业名称
#SBATCH --partition=normal                       # 分区名称 (请根据您的集群修改)
#SBATCH --nodes=1                             # 节点数量
#SBATCH --cpus-per-task=32                     # 每个任务的CPU核心数
#SBATCH --gres=gpu:4                          # GPU数量 (根据您的需求修改)
#SBATCH --mem=128G                             # 内存需求 (根据您的需求修改)
#SBATCH --time=48:00:00                       # 最大运行时间 (格式: DD-HH:MM:SS)
#SBATCH --output=/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/notebooks/logs/train_samll_batch_4gpu_w_clamp_w_reg_correct_normalize_argument.out            # 标准输出文件
#SBATCH --error=/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/notebooks/logs/train_small_batch_4gpu_w_clamp_w_reg_correct_normalize_argument.err             # 标准错误文件

# 激活conda环境
source activate diffusion


# 显示作业信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# 显示GPU信息
nvidia-smi

# 显示环境信息
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo "CUDA available:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo "CUDA device count:"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# 切换到工作目录
cd /work/home/maben/project/blue_whale_lab/projects/pareto_ebm/notebooks/scripts

# 运行训练脚本
echo "Starting training..."
python train_joint_energy_based_model_CIFAR10.py

# 训练完成后的处理
echo "Training completed!"
echo "Job finished at: $(date)"

# 可选：保存模型检查点到指定位置
# cp -r notebooks/checkpoints/CIFAR10 /path/to/backup/location/

echo "Job $SLURM_JOB_ID completed successfully!" 