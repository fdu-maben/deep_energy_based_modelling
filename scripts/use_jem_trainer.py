#!/usr/bin/env python3
"""
使用PyTorch Lightning JEM Trainer的示例
"""

# 导入训练器
from jem_lightning_trainer import train_jem_with_lightning, plot_generated_images

def main():
    print("PyTorch Lightning JEM Trainer 使用示例")
    print("=" * 50)
    
    # 训练模型
    print("开始训练JEM模型...")
    model, trainer = train_jem_with_lightning(
        max_epochs=10,           # 训练轮数
        batch_size=64,           # 批次大小
        lr=1e-4,                 # 学习率
        alpha=0.1,               # JEM正则化参数
        hidden_features=32,      # 隐藏层特征数
        use_gpu=True             # 使用GPU
    )
    
    # 生成样本
    print("生成样本...")
    generated_images = model.generate_samples(
        num_samples=16,          # 生成16个样本
        steps=2000,              # MCMC步数
        step_size=0.1            # MCMC步长
    )
    
    # 可视化生成的样本
    plot_generated_images(generated_images)
    
    print("训练和生成完成！")

if __name__ == "__main__":
    main() 