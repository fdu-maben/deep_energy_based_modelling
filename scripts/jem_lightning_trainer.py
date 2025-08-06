"""
PyTorch Lightning JEM Trainer
Joint Energy-based Model 的 PyTorch Lightning 实现
集成已有的功能，包括分类、生成和能量建模
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from typing import Optional, Dict, Any
import os
from torchmetrics.image.fid import FrechetInceptionDistance

# 从原始notebook导入的类
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class JEMClassifier(nn.Module):
    def __init__(self, hidden_features=32):
        super().__init__()
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),  # [16x16]
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  # [8x8]
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),  # [4x4]
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),  # [2x2]
            Swish(),
            nn.Flatten()
        )
        self.fc_class = nn.Linear(c_hid3 * 4, 10)

    def forward(self, x):
        features = self.cnn_layers(x)
        logits = self.fc_class(features)
        energy = torch.log(torch.sum(torch.exp(logits), dim=-1))
        return energy, logits

class Sampler:
    def __init__(self, model, img_shape, sample_size, max_len=8192):
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10):
        device = next(self.model.parameters()).device 

        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape, device=device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size - n_new), dim=0).to(device)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach()

        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

        self.examples = list(inp_imgs.cpu().chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    def conditional_sample(self, target_class, steps=60, step_size=10, num_samples=100):
        """条件采样：生成指定类别的样本（基于p(y|x)最大值）"""
        device = next(self.model.parameters()).device
        
        # 生成大量候选样本
        candidate_imgs = torch.rand((num_samples,) + self.img_shape, device=device) * 2 - 1
        
        # 条件采样
        candidate_imgs = Sampler.generate_conditional_samples(
            self.model, candidate_imgs, target_class, steps=steps, step_size=step_size
        )
        
        # 计算所有候选样本的p(y|x)值
        with torch.no_grad():
            _, logits = self.model(candidate_imgs)
            # 计算p(y|x) = softmax(logits)
            p_y_given_x = F.softmax(logits, dim=1)
            # 获取目标类别的概率
            target_probs = p_y_given_x[:, target_class]
        
        # 选择p(y|x)值最高的前10%样本
        num_top_samples = max(1, int(0.1 * num_samples))  # 至少保留1个样本
        _, top_indices = torch.topk(target_probs, num_top_samples)
        selected_imgs = candidate_imgs[top_indices]
        
        return selected_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        device = inp_imgs.device
        is_training = model.training
        model.eval()

        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        noise = torch.randn(inp_imgs.shape, device=device) 

        imgs_per_step = []

        for _ in range(steps):
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            out_imgs = -model(inp_imgs)[0]
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)

            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs

    @staticmethod
    def generate_conditional_samples(model, inp_imgs, target_class, steps=60, step_size=10, return_img_per_step=False):
        """条件采样：生成指定类别的样本（基于p(y|x)最大值）"""
        device = inp_imgs.device
        is_training = model.training
        model.eval()

        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        noise = torch.randn(inp_imgs.shape, device=device) 

        imgs_per_step = []

        for _ in range(steps):
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # 获取能量和logits
            energy, logits = model(inp_imgs)
            
            # 计算p(y|x) = softmax(logits)
            p_y_given_x = F.softmax(logits, dim=1)
            
            # 条件能量：使用目标类别的p(y|x)值作为能量
            # 我们要最大化p(y|x)，所以使用负号（因为我们要最小化能量）
            conditional_energy = -p_y_given_x[:, target_class]
            
            # 反向传播
            conditional_energy.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)

            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs

class JEMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Joint Energy-based Model (JEM)
    集成已有的JEM功能，包括分类、生成和能量建模
    """
    
    def __init__(
        self,
        hidden_features: int = 32,
        num_classes: int = 10,
        lr: float = 1e-4,
        beta1: float = 0.0,
        alpha: float = 0.1,
        sample_size: int = 64,
        max_buffer_len: int = 8192,
        mcmc_steps: int = 60,
        mcmc_step_size: float = 10.0,
        scheduler_step_size: int = 1,
        scheduler_gamma: float = 0.97,
        fid_num_samples: int = 1000,
        fid_batch_size: int = 50
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 模型
        self.model = JEMClassifier(hidden_features=hidden_features)
        
        # 采样器
        self.sampler = Sampler(
            model=self.model,
            img_shape=(1, 28, 28),
            sample_size=sample_size,
            max_len=max_buffer_len
        )
        
        # 训练参数
        self.lr = lr
        self.beta1 = beta1
        self.alpha = alpha
        self.mcmc_steps = mcmc_steps
        self.mcmc_step_size = mcmc_step_size
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # FID计算参数
        self.fid_num_samples = fid_num_samples
        self.fid_batch_size = fid_batch_size
        
        # 损失函数
        self.classification_loss = nn.CrossEntropyLoss()
        
        # FID计算器
        self.fid = FrechetInceptionDistance(feature=64, normalize=True).cpu()
        
        # 用于记录训练指标
        self.train_losses = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.test_real_images = []  # 存储测试集真实图像用于FID计算
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        real_imgs, labels = batch
        
        # 添加小噪声到真实图像
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs = real_imgs + small_noise
        real_imgs = real_imgs.clamp(min=-1.0, max=1.0)
        
        # 生成假图像
        fake_imgs = self.sampler.sample_new_exmps(
            steps=self.mcmc_steps, 
            step_size=self.mcmc_step_size
        )
        
        # 计算能量和logits
        energy_real, logits_real = self.model(real_imgs)
        energy_fake, _ = self.model(fake_imgs)
        
        # 计算JEM损失
        total_loss, reg_loss, cdiv_loss, class_loss = self.jem_loss(
            energy_real, energy_fake, logits_real, labels
        )
        
        # 记录损失
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/reg_loss', reg_loss)
        self.log('train/cdiv_loss', cdiv_loss)
        self.log('train/class_loss', class_loss)
        
        # 记录训练损失用于后续分析
        self.train_losses.append(total_loss.item())
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        val_imgs, val_labels = batch
        
        # 只进行分类任务验证
        with torch.no_grad():
            _, logits_val = self.model(val_imgs)
            predictions = torch.argmax(logits_val, dim=1)
            accuracy = (predictions == val_labels).float().mean()
            
        self.log('val/accuracy', accuracy, prog_bar=True)
        self.val_accuracies.append(accuracy.item())
        
        return accuracy
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        test_imgs, test_labels = batch
        
        # 分类任务测试
        with torch.no_grad():
            _, logits_test = self.model(test_imgs)
            predictions = torch.argmax(logits_test, dim=1)
            accuracy = (predictions == test_labels).float().mean()
            
        self.log('test/accuracy', accuracy, prog_bar=True)
        self.test_accuracies.append(accuracy.item())
        
        # 收集真实图像用于FID计算
        # 将图像从[-1,1]范围转换到[0,1]范围，并转换为3通道
        real_imgs_normalized = (test_imgs + 1) / 2  # [-1,1] -> [0,1]
        real_imgs_3ch = real_imgs_normalized.repeat(1, 3, 1, 1)  # 1通道 -> 3通道
        # 移动到CPU以用于FID计算
        real_imgs_3ch = real_imgs_3ch.cpu()
        self.test_real_images.append(real_imgs_3ch)
        
        return accuracy
    
    def jem_loss(self, energy_real, energy_fake, logits, labels):
        """JEM损失函数"""
        min_batch_size = min(energy_real.size(0), energy_fake.size(0))
        energy_real = energy_real[:min_batch_size]
        energy_fake = energy_fake[:min_batch_size]

        # 正则化损失
        reg_loss = self.alpha * (energy_real ** 2 + energy_fake ** 2).mean()
        
        # 对比散度损失
        cdiv_loss = energy_fake.mean() - energy_real.mean()

        # 分类损失
        class_loss = self.classification_loss(logits, labels)

        # 总损失
        total_loss = reg_loss + cdiv_loss + class_loss
        
        return total_loss, reg_loss, cdiv_loss, class_loss
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            betas=(self.beta1, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.scheduler_step_size, 
            gamma=self.scheduler_gamma
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/accuracy",
                "interval": "epoch"
            }
        }
    
    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        # 更新采样器中的模型引用
        self.sampler.model = self.model
        
        # 记录平均训练损失
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        self.log('train/avg_loss', avg_train_loss)
        self.train_losses = []  # 重置
        
        # 每5个epoch生成并记录样本图像
        if self.current_epoch % 5 == 0:
            self._log_generated_samples("train")
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的回调"""
        # 记录平均验证准确率
        if self.val_accuracies:
            avg_val_accuracy = sum(self.val_accuracies) / len(self.val_accuracies)
            self.log('val/avg_accuracy', avg_val_accuracy)
            self.val_accuracies = []  # 重置
        
        # 每个epoch都生成并记录样本图像到TensorBoard
        self._log_generated_samples("val")
        
        # 条件采样：随机选择一类进行条件采样
        self._log_conditional_samples("val")
    
    def on_test_epoch_end(self):
        """测试epoch结束时的回调"""
        # 记录平均测试准确率
        if self.test_accuracies:
            avg_test_accuracy = sum(self.test_accuracies) / len(self.test_accuracies)
            self.log('test/avg_accuracy', avg_test_accuracy)
            self.test_accuracies = []  # 重置
        
        # 计算FID分数
        if self.test_real_images:
            self._compute_fid_score()
            self.test_real_images = []  # 重置
        
        # 生成并记录测试阶段的样本图像到TensorBoard
        self._log_generated_samples("test")
        
        # 条件采样：随机选择一类进行条件采样
        self._log_conditional_samples("test")
    
    def _compute_fid_score(self):
        """计算FID分数（在CPU上进行）"""
        try:
            print("开始计算FID分数（CPU模式）...")
            
            # 合并所有真实图像并移动到CPU
            real_images = torch.cat(self.test_real_images, dim=0).cpu()
            
            # 重置FID计算器
            self.fid.reset()
            
            # 添加真实图像到FID计算器
            self.fid.update(real_images, real=True)
            
            # 生成假图像并添加到FID计算器
            num_batches = (self.fid_num_samples + self.fid_batch_size - 1) // self.fid_batch_size
            
            for i in range(num_batches):
                batch_size = min(self.fid_batch_size, self.fid_num_samples - i * self.fid_batch_size)
                if batch_size <= 0:
                    break
                    
                # 生成样本
                fake_imgs = self.generate_samples(
                    num_samples=batch_size, 
                    steps=1000, 
                    step_size=0.1
                )
                
                # 将图像从[-1,1]范围转换到[0,1]范围，并转换为3通道
                fake_imgs_normalized = (fake_imgs + 1) / 2  # [-1,1] -> [0,1]
                fake_imgs_3ch = fake_imgs_normalized.repeat(1, 3, 1, 1)  # 1通道 -> 3通道
                
                # 确保图像在CPU上
                fake_imgs_3ch = fake_imgs_3ch.cpu()
                
                # 添加到FID计算器
                self.fid.update(fake_imgs_3ch, real=False)
                
                print("FID计算进度: {}/{}".format(i+1, num_batches))
            
            # 计算FID分数
            fid_score = self.fid.compute()
            
            # 记录FID分数
            self.log('test/fid_score', fid_score, prog_bar=True)
            
            print("FID分数: {:.4f}".format(fid_score.item()))
            
        except Exception as e:
            print("计算FID分数时出错: {}".format(e))
    
    def _log_generated_samples(self, stage: str):
        """记录生成的样本图像到TensorBoard（在CPU上进行）"""
        try:
            # 生成样本
            generated_imgs = self.generate_samples(num_samples=16, steps=1000, step_size=0.1)
            
            # 将图像从[-1,1]范围转换到[0,1]范围
            generated_imgs_normalized = (generated_imgs + 1) / 2
            
            # 确保图像在CPU上用于TensorBoard记录
            generated_imgs_normalized = generated_imgs_normalized.cpu()
            
            # 创建图像网格
            grid = torchvision.utils.make_grid(generated_imgs_normalized, nrow=4, padding=2, normalize=False)
            
            # 记录到TensorBoard
            self.logger.experiment.add_image(
                '{}/generated_samples'.format(stage), 
                grid, 
                self.current_epoch
            )
            
            print("已记录生成样本到TensorBoard")
            
        except Exception as e:
            print("记录生成样本时出错: {}".format(e))
    
    def _log_conditional_samples(self, stage: str):
        """记录条件采样的样本图像到TensorBoard（基于p(y|x)最大值）"""
        try:
            # 随机选择一类进行条件采样
            target_class = random.randint(0, 9)  # MNIST有10个类别 (0-9)
            
            # 生成条件采样样本（使用更多候选样本以获得更好的质量）
            conditional_imgs = self.generate_conditional_samples(
                target_class=target_class,
                num_samples=16, 
                steps=1000, 
                step_size=0.1,
                num_candidates=200  # 生成200个候选样本，选择前10%
            )
            
            # 将图像从[-1,1]范围转换到[0,1]范围
            conditional_imgs_normalized = (conditional_imgs + 1) / 2
            
            # 确保图像在CPU上用于TensorBoard记录
            conditional_imgs_normalized = conditional_imgs_normalized.cpu()
            
            # 创建图像网格
            grid = torchvision.utils.make_grid(conditional_imgs_normalized, nrow=4, padding=2, normalize=False)
            
            # 记录到TensorBoard
            self.logger.experiment.add_image(
                '{}/conditional_samples_class_{}'.format(stage, target_class), 
                grid, 
                self.current_epoch
            )
            
            print("已记录类别{}的条件采样样本到TensorBoard（基于p(y|x)最大值）".format(target_class))
            
        except Exception as e:
            print("记录条件采样样本时出错: {}".format(e))
    
    def generate_samples(self, num_samples: int = 16, steps: int = 2000, step_size: float = 0.1):
        """生成样本（返回CPU上的张量）"""
        self.eval()
        with torch.no_grad():
            generated_imgs = self.sampler.sample_new_exmps(steps=steps, step_size=step_size)
            generated_imgs = generated_imgs[:num_samples].detach()
            
        return generated_imgs.cpu()
    
    def generate_conditional_samples(self, target_class: int, num_samples: int = 16, steps: int = 2000, step_size: float = 0.1, num_candidates: int = 100):
        """条件采样：生成指定类别的样本（基于p(y|x)最大值，返回CPU上的张量）"""
        self.eval()
        with torch.no_grad():
            # 生成候选样本并选择最佳样本
            generated_imgs = self.sampler.conditional_sample(
                target_class, 
                steps=steps, 
                step_size=step_size,
                num_samples=num_candidates
            )
            # 取前num_samples个样本
            generated_imgs = generated_imgs[:num_samples].detach()
            
        return generated_imgs.cpu()

class JEMDataModule(pl.LightningDataModule):
    """JEM数据模块"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(
                root=self.data_dir, 
                train=True, 
                transform=transform, 
                download=True
            )
            self.val_dataset = datasets.MNIST(
                root=self.data_dir, 
                train=False, 
                transform=transform, 
                download=True
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(
                root=self.data_dir, 
                train=False, 
                transform=transform, 
                download=True
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,  # 加速数据传输到GPU
            persistent_workers=True  # 保持worker进程存活
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,  # 加速数据传输到GPU
            persistent_workers=True  # 保持worker进程存活
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,  # 加速数据传输到GPU
            persistent_workers=True  # 保持worker进程存活
        )

def train_jem_with_lightning(
    max_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    alpha: float = 0.1,
    hidden_features: int = 32,
    data_dir: str = "/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/project/datasets",
    checkpoint_dir: str = "lightning_logs",
    use_gpu: bool = True,
    experiment_name: str = "jem_training",
    fid_num_samples: int = 1000,
    fid_batch_size: int = 50,
    num_gpus: int = 4,
    num_workers: int = 4
):
    """
    使用PyTorch Lightning训练JEM模型（DDP分布式训练）
    
    Args:
        max_epochs: 最大训练轮数
        batch_size: 每个GPU的批次大小（总批次大小 = batch_size * num_gpus）
        lr: 学习率
        alpha: JEM损失中的正则化参数
        hidden_features: 隐藏层特征数
        data_dir: 数据目录
        checkpoint_dir: 检查点保存目录
        use_gpu: 是否使用GPU
        experiment_name: TensorBoard实验名称
        fid_num_samples: FID计算使用的样本数量
        fid_batch_size: FID计算的批次大小
        num_gpus: 使用的GPU数量
        num_workers: 每个GPU的数据加载器worker数量
    """
    
    # 创建数据模块
    data_module = JEMDataModule(
        data_dir=data_dir,
        batch_size=batch_size,  # 这是每个GPU的批次大小
        num_workers=num_workers
    )
    
    # 创建模型
    model = JEMLightningModule(
        hidden_features=hidden_features,
        num_classes=10,
        lr=lr,
        alpha=alpha,
        fid_num_samples=fid_num_samples,
        fid_batch_size=fid_batch_size
    )
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='jem-{epoch:02d}-{val_accuracy:.4f}',
            monitor='val/accuracy',
            mode='max',
            save_top_k=3,
            save_last=True
        )
    ]
    
    # 设置TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=checkpoint_dir,
        name=experiment_name
    )
    
    # 创建训练器（DDP配置）
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=num_gpus,
        strategy='ddp',  # 明确指定使用DDP策略
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,  # 每个epoch结束时验证一次
        gradient_clip_val=1.0,    # 梯度裁剪
        accumulate_grad_batches=1,
        precision=32,
        sync_batchnorm=True,  # 在DDP模式下同步批归一化
        deterministic=False,  # 为了性能，关闭确定性模式
        enable_progress_bar=True,  # 启用进度条
        enable_model_summary=True,  # 启用模型摘要
        enable_checkpointing=True,  # 启用检查点保存
        reload_dataloaders_every_n_epochs=0  # 不重新加载数据加载器
    )
    
    # 开始训练
    print("开始训练JEM模型（DDP分布式训练）...")
    print("GPU数量: {}, 每个GPU批次大小: {}, 总批次大小: {}".format(
        num_gpus, batch_size, batch_size * num_gpus
    ))
    trainer.fit(model, data_module)
    
    print("训练完成！")
    
    # 运行测试阶段（包括FID计算）
    print("开始测试阶段（包括FID计算）...")
    trainer.test(model, data_module)
    
    print("测试完成！")
    
    # 返回训练好的模型
    return model, trainer

def plot_generated_images(images, num_cols=4):
    """可视化生成的图像"""
    num_images = images.size(0)
    num_rows = (num_images + num_cols - 1) // num_cols

    images = (images + 1) / 2 

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i].squeeze(0).numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 训练模型
    model, trainer = train_jem_with_lightning(
        max_epochs=10,
        batch_size=32,  # 每个GPU的批次大小，总批次大小为16*4=64
        lr=1e-4,
        alpha=0.1,
        hidden_features=32,
        experiment_name="jem_experiment_001",
        fid_num_samples=500,  # FID计算使用的样本数量
        fid_batch_size=50,    # FID计算的批次大小
        num_gpus=4,           # 使用4个GPU
        num_workers=4         # 每个GPU使用4个worker
    )
    
    print("训练和测试完成！现在生成样本进行可视化...")
    
    # 生成无条件样本
    print("生成无条件样本...")
    generated_images = model.generate_samples(num_samples=16, steps=2000, step_size=0.1)
    
    # 可视化无条件生成的样本
    plot_generated_images(generated_images)
    
    # 生成条件样本（为每个类别生成样本）
    print("生成条件样本（基于p(y|x)最大值）...")
    for target_class in range(10):  # MNIST有10个类别
        print("生成类别{}的条件样本...".format(target_class))
        conditional_images = model.generate_conditional_samples(
            target_class=target_class, 
            num_samples=16, 
            steps=2000, 
            step_size=0.1,
            num_candidates=200  # 生成200个候选样本，选择前10%
        )
        
        # 可视化条件生成的样本
        plot_generated_images(conditional_images)
    
    # 保存模型
    torch.save(model.state_dict(), "jem_model_final.pth")
    print("模型已保存到 jem_model_final.pth") 