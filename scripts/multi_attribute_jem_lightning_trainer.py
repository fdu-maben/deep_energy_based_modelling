"""
Multi-Attribute Joint Energy-based Model (JEM) 的 PyTorch Lightning 实现
基于references/train_joint.py和CelebA数据集
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
from typing import Optional, Dict, Any, List, Tuple
import os
from torch.nn.utils import spectral_norm, clip_grad_norm_
from torchvision.utils import save_image
import torch.distributions as dists

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class WSConv2d(nn.Conv2d):
    """Weight Standardization Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=1, keepdim=True).std(dim=2,
               keepdim=True).std(dim=3, keepdim=True) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class CondResBlock(nn.Module):
    """条件残差块，用于CelebA模型"""
    def __init__(self, filters=64, latent_dim=64, im_size=64, classes=40, 
                 downsample=True, rescale=True, norm=True, spec_norm=False):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample

        if filters <= 128:
            self.bn1 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters)

        if not norm:
            self.bn1 = None

        if spec_norm:
            self.conv1 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv1 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        if not norm:
            self.bn2 = None

        if spec_norm:
            self.conv2 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv2 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.2)

        # 条件映射
        self.latent_map = nn.Linear(classes, 2*filters)
        self.latent_map_2 = nn.Linear(classes, 2*filters)

        self.relu = torch.nn.ReLU(inplace=True)
        self.act = nn.SiLU(inplace=True)

        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
                self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
                self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, y):
        if y is not None:
            latent_map = self.latent_map(y).view(-1, 2*self.filters, 1, 1)
            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]
        else:
            gain = bias = None

        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)

        if y is not None:
            latent_map = self.latent_map_2(y).view(-1, 2*self.filters, 1, 1)
            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv2(x)

        if self.bn2 is not None:
            x = self.bn2(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)

        x_out = x

        if self.downsample:
            x_out = self.conv_downsample(x_out)
            x_out = self.act(self.avg_pool(x_out))

        return x_out 

class Self_Attn(nn.Module):
    """自注意力模块"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        if self.activation:
            out = self.activation(out)

        return out, attention

class CelebAModel(nn.Module):
    """CelebA多属性联合能量模型"""
    def __init__(self, n_f=32, img_size=64, num_attributes=40, multiscale=False, 
                 self_attn=False, norm=True, spec_norm=False):
        super(CelebAModel, self).__init__()
        self.act = nn.SiLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cond = True

        self.n_f = n_f
        self.img_size = img_size
        self.num_attributes = num_attributes
        self.multiscale = multiscale
        self.self_attn = self_attn
        self.norm = norm
        self.spec_norm = spec_norm

        self.init_main_model()

        if multiscale:
            self.init_mid_model()
            self.init_small_model()

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = Downsample(channels=3)
        self.heir_weight = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))

    def init_main_model(self):
        filter_dim = self.n_f
        latent_dim = self.n_f
        im_size = self.img_size

        self.conv1 = nn.Conv2d(3, filter_dim // 2, kernel_size=3, stride=1, padding=1)

        self.res_1a = CondResBlock(
            filters=filter_dim // 2,
            latent_dim=latent_dim,
            im_size=im_size,
            classes=self.num_attributes,
            downsample=True,
            norm=self.norm,
            spec_norm=self.spec_norm
        )
        self.res_1b = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=False,
            classes=self.num_attributes,
            norm=self.norm,
            spec_norm=self.spec_norm
        )

        self.res_2a = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=True,
            rescale=False,
            classes=self.num_attributes,
            norm=self.norm,
            spec_norm=self.spec_norm
        )
        self.res_2b = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=True,
            classes=self.num_attributes,
            norm=self.norm,
            spec_norm=self.spec_norm
        )

        self.res_3a = CondResBlock(
            filters=2 * filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=False,
            classes=self.num_attributes,
            norm=self.norm,
            spec_norm=self.spec_norm
        )
        self.res_3b = CondResBlock(
            filters=2 * filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=True,
            classes=self.num_attributes,
            norm=self.norm,
            spec_norm=self.spec_norm
        )

        self.res_4a = CondResBlock(
            filters=4 * filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=False,
            classes=self.num_attributes,
            norm=self.norm,
            spec_norm=self.spec_norm
        )
        self.res_4b = CondResBlock(
            filters=4 * filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=True,
            classes=self.num_attributes,
            norm=self.norm,
            spec_norm=self.spec_norm
        )

        if self.self_attn:
            self.self_attn_layer = Self_Attn(4 * filter_dim, self.act)

        self.energy_map = nn.Linear(filter_dim*8, 1)
        self.label_map = nn.Linear(filter_dim*8, self.num_attributes)

    def init_mid_model(self):
        filter_dim = self.n_f
        latent_dim = self.n_f
        im_size = self.img_size

        self.mid_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.mid_res_1a = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=True,
            rescale=False,
            classes=self.num_attributes
        )
        self.mid_res_1b = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=False,
            classes=self.num_attributes
        )

        self.mid_res_2a = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=True,
            rescale=False,
            classes=self.num_attributes
        )
        self.mid_res_2b = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=True,
            classes=self.num_attributes
        )

        self.mid_res_3a = CondResBlock(
            filters=2 * filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=False,
            classes=self.num_attributes
        )
        self.mid_res_3b = CondResBlock(
            filters=2 * filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=True,
            classes=self.num_attributes
        )

        self.mid_energy_map = nn.Linear(filter_dim * 4, 1)
        self.avg_pool = Downsample(channels=3)

    def init_small_model(self):
        filter_dim = self.n_f
        latent_dim = self.n_f
        im_size = self.img_size

        self.small_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.small_res_1a = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=True,
            rescale=False,
            classes=self.num_attributes
        )
        self.small_res_1b = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=False,
            classes=self.num_attributes
        )

        self.small_res_2a = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            downsample=True,
            rescale=False,
            classes=self.num_attributes
        )
        self.small_res_2b = CondResBlock(
            filters=filter_dim,
            latent_dim=latent_dim,
            im_size=im_size,
            rescale=True,
            classes=self.num_attributes
        )

        self.small_energy_map = nn.Linear(filter_dim * 2, 1)

    def main_model(self, x, latent):
        x = self.act(self.conv1(x))

        x = self.res_1a(x, latent)
        x = self.res_1b(x, latent)

        x = self.res_2a(x, latent)
        x = self.res_2b(x, latent)

        x = self.res_3a(x, latent)
        x = self.res_3b(x, latent)

        if self.self_attn:
            x, _ = self.self_attn_layer(x)

        x = self.res_4a(x, latent)
        x = self.res_4b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)
        
        energy = self.energy_map(x)
        logits = self.label_map(x)

        return energy, logits

    def mid_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x, latent)
        x = self.mid_res_1b(x, latent)

        x = self.mid_res_2a(x, latent)
        x = self.mid_res_2b(x, latent)

        x = self.mid_res_3a(x, latent)
        x = self.mid_res_3b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)
        energy = self.mid_energy_map(x)

        return energy

    def small_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x, latent)
        x = self.small_res_1b(x, latent)

        x = self.small_res_2a(x, latent)
        x = self.small_res_2b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)
        energy = self.small_energy_map(x)

        return energy

    def forward(self, x, latent=None):
        if not self.cond:
            latent = None

        energy, logits = self.main_model(x, latent)

        if self.multiscale:
            large_energy = energy
            mid_energy = self.mid_model(x, latent)
            small_energy = self.small_model(x, latent)
            energy = torch.cat([small_energy, mid_energy, large_energy], dim=-1)

        return energy, logits 

class ReplayBuffer:
    """重放缓冲区"""
    def __init__(self, max_size, example_sample):
        self.max_size = max_size
        self.buffer = []
        self.example_sample = example_sample

    def add(self, x):
        if len(self.buffer) < self.max_size:
            self.buffer.append(x.detach().cpu())
        else:
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer[idx] = x.detach().cpu()

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        samples = [self.buffer[i] for i in indices]
        return torch.stack(samples)

    def __len__(self):
        return len(self.buffer)

class Sampler:
    """MCMC采样器"""
    def __init__(self, model, img_shape, sample_size, max_len=8192):
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.replay_buffer = ReplayBuffer(max_len, torch.randn(1, *img_shape))

    def sample_new_exmps(self, steps=60, step_size=10):
        init_samples = torch.randn(self.sample_size, *self.img_shape).to(next(self.model.parameters()).device)
        init_samples.requires_grad_()
        
        for _ in range(steps):
            energy, _ = self.model(init_samples)
            energy.sum().backward()
            
            with torch.no_grad():
                init_samples.data += step_size * init_samples.grad.data
                init_samples.grad.zero_()
                
        return init_samples.detach()

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        imgs_per_step = []
        x_k = inp_imgs.clone().detach()
        x_k.requires_grad_()
        
        for k in range(steps):
            energy, _ = model(x_k)
            energy.sum().backward()
            
            with torch.no_grad():
                x_k.data += step_size * x_k.grad.data
                x_k.grad.zero_()
                
            if return_img_per_step and k % 20 == 0:
                imgs_per_step.append(x_k.detach().clone())
        
        if return_img_per_step:
            return x_k.detach(), imgs_per_step
        else:
            return x_k.detach()

class MultiAttributeJEMLightningModule(pl.LightningModule):
    """多属性联合能量模型的PyTorch Lightning模块"""
    
    def __init__(
        self,
        n_f: int = 32,
        img_size: int = 64,
        num_attributes: int = 40,
        lr: float = 1e-4,
        beta1: float = 0.0,
        alpha: float = 0.1,
        sample_size: int = 64,
        max_buffer_len: int = 8192,
        mcmc_steps: int = 60,
        mcmc_step_size: float = 10.0,
        multiscale: bool = False,
        self_attn: bool = False,
        norm: bool = True,
        spec_norm: bool = False,
        scheduler_step_size: int = 1,
        scheduler_gamma: float = 0.97
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 模型参数
        self.n_f = n_f
        self.img_size = img_size
        self.num_attributes = num_attributes
        self.lr = lr
        self.beta1 = beta1
        self.alpha = alpha
        self.sample_size = sample_size
        self.max_buffer_len = max_buffer_len
        self.mcmc_steps = mcmc_steps
        self.mcmc_step_size = mcmc_step_size
        self.multiscale = multiscale
        self.self_attn = self_attn
        self.norm = norm
        self.spec_norm = spec_norm
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # 初始化模型
        self.model = CelebAModel(
            n_f=n_f,
            img_size=img_size,
            num_attributes=num_attributes,
            multiscale=multiscale,
            self_attn=self_attn,
            norm=norm,
            spec_norm=spec_norm
        )
        
        # 初始化采样器
        self.sampler = Sampler(
            self.model,
            img_shape=(3, img_size, img_size),
            sample_size=sample_size,
            max_len=max_buffer_len
        )
        
        # 损失函数
        self.ce_loss = nn.BCEWithLogitsLoss()
        
        # 训练统计
        self.train_step_count = 0
        
    def forward(self, x, latent=None):
        return self.model(x, latent)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        
        # 生成负样本
        if self.train_step_count % 5 == 0:
            x_fake = self.sampler.sample_new_exmps(
                steps=self.mcmc_steps,
                step_size=self.mcmc_step_size
            )
            self.sampler.replay_buffer.add(x_fake)
        else:
            buffer_samples = self.sampler.replay_buffer.sample(self.sample_size)
            if buffer_samples is not None:
                x_fake = buffer_samples.to(x.device)
            else:
                x_fake = torch.randn_like(x)
        
        # 前向传播
        energy_real, logits_real = self.model(x)
        energy_fake, logits_fake = self.model(x_fake)
        
        # 计算损失
        loss = self.jem_loss(energy_real, energy_fake, logits_real, labels)
        
        # 记录指标
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_energy_real', energy_real.mean(), prog_bar=True)
        self.log('train_energy_fake', energy_fake.mean(), prog_bar=True)
        
        self.train_step_count += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        
        energy_real, logits_real = self.model(x)
        
        # 计算分类准确率
        pred_labels = (logits_real > 0).float()
        accuracy = (pred_labels == labels).float().mean()
        
        # 计算F1分数
        f1_score = self.calculate_f1_score(labels, pred_labels)
        
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_f1_score', f1_score, prog_bar=True)
        self.log('val_energy_real', energy_real.mean(), prog_bar=True)
        
        return {'val_accuracy': accuracy, 'val_f1_score': f1_score}
    
    def jem_loss(self, energy_real, energy_fake, logits, labels):
        """JEM损失函数"""
        # 能量损失
        energy_loss = energy_real.mean() - energy_fake.mean()
        
        # 分类损失
        classification_loss = self.ce_loss(logits, labels.float())
        
        # 总损失
        total_loss = energy_loss + self.alpha * classification_loss
        
        return total_loss
    
    def calculate_f1_score(self, y_true, y_pred):
        """计算F1分数"""
        tp = (y_true == y_pred) * (y_true == 1)
        fp = (y_pred == 1) * (y_true == 0)
        fn = (y_pred == 0) * (y_true == 1)
        
        precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
        recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def on_train_epoch_end(self):
        """训练轮次结束时的回调"""
        # 生成样本用于可视化
        if self.current_epoch % 5 == 0:
            self.generate_and_log_samples()
    
    def on_validation_epoch_end(self):
        """验证轮次结束时的回调"""
        pass
    
    def generate_and_log_samples(self):
        """生成样本并记录到日志"""
        with torch.no_grad():
            # 生成无条件样本
            x_fake = self.sampler.sample_new_exmps(
                steps=self.mcmc_steps,
                step_size=self.mcmc_step_size
            )
            
            # 保存样本图像
            if self.logger:
                grid = torchvision.utils.make_grid(x_fake[:16], nrow=4, normalize=True)
                self.logger.experiment.add_image(
                    f'generated_samples_epoch_{self.current_epoch}',
                    grid,
                    self.current_epoch
                )
    
    def generate_samples(self, num_samples: int = 16, steps: int = 2000, step_size: float = 0.1):
        """生成样本"""
        self.model.eval()
        with torch.no_grad():
            samples = self.sampler.sample_new_exmps(
                steps=steps,
                step_size=step_size
            )
        self.model.train()
        return samples[:num_samples]

class CelebADataModule(pl.LightningDataModule):
    """CelebA数据模块"""
    
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.CelebA(
                root=self.data_dir,
                split='train',
                transform=self.transform,
                target_type='attr',
                download=False
            )
            self.val_dataset = datasets.CelebA(
                root=self.data_dir,
                split='valid',
                transform=self.transform,
                target_type='attr',
                download=False
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CelebA(
                root=self.data_dir,
                split='test',
                transform=self.transform,
                target_type='attr',
                download=False
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 

def train_multi_attribute_jem_with_lightning(
    max_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    alpha: float = 0.1,
    n_f: int = 32,
    img_size: int = 64,
    num_attributes: int = 40,
    data_dir: str = "/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/project/datasets",
    checkpoint_dir: str = "lightning_logs",
    use_gpu: bool = True,
    multiscale: bool = False,
    self_attn: bool = False,
    norm: bool = True,
    spec_norm: bool = False
):
    """训练多属性联合能量模型"""
    
    # 创建数据模块
    data_module = CelebADataModule(
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    # 创建模型
    model = MultiAttributeJEMLightningModule(
        n_f=n_f,
        img_size=img_size,
        num_attributes=num_attributes,
        lr=lr,
        alpha=alpha,
        multiscale=multiscale,
        self_attn=self_attn,
        norm=norm,
        spec_norm=spec_norm
    )
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='multi_attribute_jem-{epoch:02d}-{val_f1_score:.3f}',
        monitor='val_f1_score',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_f1_score',
        patience=10,
        mode='max'
    )
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=checkpoint_dir,
        name='multi_attribute_jem'
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if use_gpu and torch.cuda.is_available() else 'cpu',
        devices=4,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        precision=16 if use_gpu else 32
    )
    
    # 开始训练
    trainer.fit(model, data_module)
    
    return model, trainer

def plot_generated_samples(images, num_cols=4, save_path=None):
    """可视化生成的样本"""
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols
        
        img = images[i].cpu().detach()
        img = (img + 1) / 2  # 反归一化
        img = torch.clamp(img, 0, 1)
        
        axes[row, col].imshow(img.permute(1, 2, 0))
        axes[row, col].axis('off')
        axes[row, col].set_title('Sample {}'.format(i+1))
    
    # 隐藏多余的子图
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # 使用示例
    print("开始训练多属性联合能量模型...")
    
    model, trainer = train_multi_attribute_jem_with_lightning(
        max_epochs=10,
        batch_size=32,
        lr=1e-4,
        alpha=0.1,
        n_f=32,
        img_size=64,
        num_attributes=40,
        multiscale=False,
        self_attn=False,
        norm=True,
        spec_norm=False
    )
    
    # 生成样本
    print("生成样本...")
    generated_samples = model.generate_samples(num_samples=16)
    
    # 可视化样本
    plot_generated_samples(generated_samples, save_path="generated_samples.png")
    
    print("训练完成！") 