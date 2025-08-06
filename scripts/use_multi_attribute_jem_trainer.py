#!/usr/bin/env python3
"""
使用Multi-Attribute Joint Energy-based Model (JEM) PyTorch Lightning Trainer的示例
"""

# 导入训练器
from multi_attribute_jem_lightning_trainer import (
    train_multi_attribute_jem_with_lightning, 
    plot_generated_samples,
    MultiAttributeJEMLightningModule,
    CelebADataModule
)

def main():
    print("Multi-Attribute Joint Energy-based Model (JEM) 使用示例")
    print("=" * 60)
    
    # 训练模型
    print("开始训练多属性联合能量模型...")
    model, trainer = train_multi_attribute_jem_with_lightning(
        max_epochs=10,           # 训练轮数
        batch_size=32,           # 批次大小
        lr=1e-4,                 # 学习率
        alpha=0.1,               # JEM正则化参数
        n_f=32,                  # 特征维度
        img_size=64,             # 图像尺寸
        num_attributes=40,       # 属性数量
        multiscale=False,        # 是否使用多尺度
        self_attn=False,         # 是否使用自注意力
        norm=True,               # 是否使用归一化
        spec_norm=False          # 是否使用谱归一化
    )
    
    # 生成样本
    print("生成样本...")
    generated_samples = model.generate_samples(
        num_samples=16,          # 生成16个样本
        steps=2000,              # MCMC步数
        step_size=0.1            # MCMC步长
    )
    
    # 可视化生成的样本
    plot_generated_samples(generated_samples, save_path="multi_attribute_generated_samples.png")
    
    print("训练完成！")
    print("生成的样本已保存到 multi_attribute_generated_samples.png")

def test_data_loading():
    """测试数据加载"""
    print("测试CelebA数据加载...")
    
    data_module = CelebADataModule(
        data_dir="/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/project/datasets",
        batch_size=4
    )
    
    data_module.setup()
    
    # 获取一个批次的数据
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    images, labels = batch
    print(f"图像形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签范围: {labels.min().item():.2f} - {labels.max().item():.2f}")
    
    print("数据加载测试完成！")

def test_model():
    """测试模型"""
    print("测试多属性JEM模型...")
    
    model = MultiAttributeJEMLightningModule(
        n_f=16,                  # 较小的特征维度用于测试
        img_size=64,
        num_attributes=40,
        lr=1e-4,
        alpha=0.1
    )
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    labels = torch.randint(0, 2, (batch_size, 40)).float()
    
    # 前向传播
    energy, logits = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"能量形状: {energy.shape}")
    print(f"逻辑形状: {logits.shape}")
    
    print("模型测试完成！")

if __name__ == "__main__":
    # 运行测试
    test_data_loading()
    test_model()
    
    # 运行主训练
    main() 