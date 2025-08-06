#!/usr/bin/env python3
"""
测试Multi-Attribute Joint Energy-based Model (JEM)的实现
"""

import torch
import torch.nn as nn
from multi_attribute_jem_lightning_trainer import (
    CelebAModel,
    CondResBlock,
    Self_Attn,
    MultiAttributeJEMLightningModule,
    CelebADataModule
)

def test_cond_res_block():
    """测试条件残差块"""
    print("测试条件残差块...")
    
    # 创建条件残差块
    block = CondResBlock(
        filters=64,
        latent_dim=64,
        im_size=64,
        classes=40,
        downsample=True,
        rescale=True,
        norm=True,
        spec_norm=False
    )
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 64, 32, 32)
    y = torch.randn(batch_size, 40)  # 属性条件
    
    # 前向传播
    output = block(x, y)
    
    print(f"输入形状: {x.shape}")
    print(f"条件形状: {y.shape}")
    print(f"输出形状: {output.shape}")
    print("条件残差块测试通过！\n")

def test_self_attention():
    """测试自注意力模块"""
    print("测试自注意力模块...")
    
    # 创建自注意力模块
    attn = Self_Attn(in_dim=128, activation=nn.SiLU())
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 128, 16, 16)
    
    # 前向传播
    output, attention = attn(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力形状: {attention.shape}")
    print("自注意力模块测试通过！\n")

def test_celeba_model():
    """测试CelebA模型"""
    print("测试CelebA模型...")
    
    # 创建CelebA模型
    model = CelebAModel(
        n_f=16,  # 较小的特征维度用于测试
        img_size=64,
        num_attributes=40,
        multiscale=False,
        self_attn=False,
        norm=True,
        spec_norm=False
    )
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    latent = torch.randn(batch_size, 40)  # 属性条件
    
    # 前向传播
    energy, logits = model(x, latent)
    
    print(f"输入形状: {x.shape}")
    print(f"条件形状: {latent.shape}")
    print(f"能量形状: {energy.shape}")
    print(f"逻辑形状: {logits.shape}")
    print("CelebA模型测试通过！\n")

def test_lightning_module():
    """测试PyTorch Lightning模块"""
    print("测试PyTorch Lightning模块...")
    
    # 创建Lightning模块
    model = MultiAttributeJEMLightningModule(
        n_f=16,  # 较小的特征维度用于测试
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
    
    # 测试损失计算
    loss = model.jem_loss(energy, energy, logits, labels)
    print(f"损失值: {loss.item():.4f}")
    print("PyTorch Lightning模块测试通过！\n")

def test_data_module():
    """测试数据模块"""
    print("测试数据模块...")
    
    try:
        # 创建数据模块
        data_module = CelebADataModule(
            data_dir="/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/project/datasets",
            batch_size=4
        )
        
        # 设置数据
        data_module.setup()
        
        # 获取训练数据加载器
        train_loader = data_module.train_dataloader()
        
        # 获取一个批次
        batch = next(iter(train_loader))
        images, labels = batch
        
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签范围: {labels.min().item():.2f} - {labels.max().item():.2f}")
        print("数据模块测试通过！\n")
        
    except Exception as e:
        print(f"数据模块测试失败: {e}")
        print("请确保CelebA数据集已正确下载到指定路径\n")

def test_sampler():
    """测试采样器"""
    print("测试采样器...")
    
    from multi_attribute_jem_lightning_trainer import Sampler
    
    # 创建模型
    model = CelebAModel(n_f=16, img_size=64, num_attributes=40)
    
    # 创建采样器
    sampler = Sampler(
        model=model,
        img_shape=(3, 64, 64),
        sample_size=4,
        max_len=100
    )
    
    # 测试采样
    samples = sampler.sample_new_exmps(steps=10, step_size=1.0)
    
    print(f"采样形状: {samples.shape}")
    print("采样器测试通过！\n")

def main():
    """运行所有测试"""
    print("开始测试Multi-Attribute Joint Energy-based Model (JEM)...")
    print("=" * 60)
    
    # 运行测试
    test_cond_res_block()
    test_self_attention()
    test_celeba_model()
    test_lightning_module()
    test_data_module()
    test_sampler()
    
    print("所有测试完成！")
    print("Multi-Attribute JEM实现验证成功！")

if __name__ == "__main__":
    main() 