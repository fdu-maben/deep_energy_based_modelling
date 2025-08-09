import os
import json
import math 
import numpy as np
import random
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST,CIFAR10
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import argparse
from wideresnet import *

DATASET_PATH = "/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/datasets"
CHECKPOINT_PATH = "/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/notebooks/checkpoints"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CNNModel(nn.Module):

    def __init__(self, hidden_features=32, out_dim=10, **kwargs):
        super().__init__()

        self.f = Wide_ResNet(depth=28, widen_factor=2, norm=None, dropout_rate=0.0)
        self.energy_output = nn.Linear(self.f.last_dim, out_dim)

    def forward(self, x):
        logits = self.energy_output(self.f(x))
        energy = torch.logsumexp(logits, dim=-1)
        return energy, logits

class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape, batch_size, max_len=10000, alpha=0.01, lr=1e-4, beta1=0.9, beta2=0.999 ,weight_decay=0.0, warmup_iters=1000,  decay_epoch=50, decay_rate=0.3, seed=42, **CNN_args):
        super().__init__()

        self.save_hyperparameters()
        self.img_shape = img_shape
        self.sample_size = batch_size
        self.max_len = max_len
        self.cnn = CNNModel(**CNN_args)
        self.example_input_array = torch.zeros(1, *img_shape)
        self.register_buffer("examples", torch.cat([(torch.rand((1,)+img_shape)*2-1) for _ in range(self.max_len)],dim=0))

    def forward(self, x):
        energy, logits = self.cnn(x)
        return energy, logits

    def configure_optimizers(self):
        def lr_lambda(epoch):
            return pow(self.hparams.decay_rate, epoch // self.hparams.decay_epoch)
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        #optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr)
        #warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.lr * 1.0/self.hparams.warmup_iters, end_factor=1.0, total_iters=self.hparams.warmup_iters)
        #decay_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        #return [optimizer],[{"scheduler" : decay_scheduler,"interval" : "epoch","frequency" : 1},{"scheduler" : warmup_scheduler,"interval" : "step","frequency" : 1}]
        return optimizer

    def training_step(self, batch, batch_idx):
        real_imgs, label = batch

        fake_imgs = self.sample_new_exmps(steps=20, step_size=1)
        fake_imgs = fake_imgs.to(self.device)

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        energy, logits = self.cnn(inp_imgs)
        real_out, fake_out = energy.chunk(2, dim=0)
        real_logits, _ = logits.chunk(2, dim=0)

        cdiv_loss = fake_out.mean() - real_out.mean()
        classification_loss = nn.CrossEntropyLoss()(real_logits,label)
        loss = cdiv_loss + classification_loss

        self.log('loss', loss, sync_dist=True)
        self.log('loss_contrastive_divergence', cdiv_loss, sync_dist=True)
        self.log('loss_classify',classification_loss, sync_dist=True)
        self.log('metrics_avg_real', real_out.mean(), sync_dist=True)
        self.log('metrics_avg_fake', fake_out.mean(), sync_dist=True)

        if loss.abs().item() > 1e8:
            1 / 0
            
        if self.global_step <= self.hparams.warmup_iters:
            self.optimizers().optimizer.param_groups[0]['lr'] = self.hparams.lr * self.global_step / float(self.hparams.warmup_iters)
        return loss

    def validation_step(self, batch, batch_idx):
        real_imgs, label = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        energy, logits = self.cnn(inp_imgs)
        real_out, fake_out = energy.chunk(2, dim=0)
        real_logits, _ = logits.chunk(2, dim=0)

        preds = torch.argmax(real_logits, dim=1)
        correct = (preds == label).sum().item()
        acc = correct / label.size(0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log('val_accuracy', acc, sync_dist=True)
        self.log('val_contrastive_divergence', cdiv, sync_dist=True)
        self.log('val_fake_out', fake_out.mean(), sync_dist=True)
        self.log('val_real_out', real_out.mean(), sync_dist=True)

    def sample_new_exmps(self, steps=20, step_size=1):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        index = np.random.randint(0,self.max_len,self.sample_size)

        old_imgs = self.examples[index]
        rand_imgs = torch.rand((self.sample_size,) + self.img_shape, device=old_imgs.device) * 2 - 1
        choose_random = (torch.rand(self.sample_size) < 0.05).float()[:, None, None, None].to(old_imgs.device)
        inp_imgs = choose_random * rand_imgs + (1 - choose_random) * old_imgs
        inp_imgs = inp_imgs.detach()

        # Perform MCMC sampling
        inp_imgs = self.generate_samples(inp_imgs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples[index] = inp_imgs
        return inp_imgs

    def generate_samples(self, inp_imgs, steps=20, step_size=1, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        is_training = self.cnn.training
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad = False
        inp_imgs = inp_imgs.to(self.device)
        inp_imgs.requires_grad = True

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        noise = torch.randn(inp_imgs.shape, device=self.device)

        imgs_per_step = []

        for _ in range(steps):

            noise.normal_(0, 1)
            inp_imgs.data.add_(0.01 * noise.data)

            out_imgs = -self.cnn(inp_imgs)[0]
            out_imgs.sum().backward()

            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        for p in self.cnn.parameters():
            p.requires_grad = True
        self.cnn.train(is_training)

        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs

    def generate_conditional_samples(self, inp_imgs, steps=20, step_size=1, conditional_index=0,return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        is_training = self.cnn.training
        self.cnn.eval()
        for p in self.cnn.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad=True
        inp_imgs = inp_imgs.to(self.device)

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        noise = torch.randn(inp_imgs.shape, device=self.device)

        imgs_per_step = []

        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 1)
            inp_imgs.data.add_(0.01*noise.data)

            # Part 2: calculate gradients for the current input.
            out_imgs = -self.cnn(inp_imgs)[1][...,conditional_index]
            out_imgs.sum().backward()

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in self.cnn.parameters():
            p.requires_grad = True
        self.cnn.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs

    '''
    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.decay_epoch == 0:
            for param_group in self.optimizers().optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.hparams.decay_rate
    '''

class RestartTrainingCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.callback_metrics['loss'].abs().item() > 1e8:
            trainer.should_stop = True

class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         
        self.vis_steps = vis_steps           
        self.num_steps = num_steps           
        self.every_n_epochs = every_n_epochs 

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            
            imgs_per_step = self.generate_imgs(pl_module)
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                grid = torchvision.utils.make_grid(imgs_to_plot.clamp_(min=-1.0, max=1.0), nrow=imgs_to_plot.shape[0], normalize=True)
                trainer.logger.experiment.add_image("generation_{}".format(i), grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True) 
        imgs_per_step = pl_module.generate_samples(start_imgs, steps=self.num_steps, step_size=1, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step

class ConditionalGenerateCallback(pl.Callback):

    def __init__(self, batch_size=1, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         # Number of images to generate
        self.vis_steps = vis_steps           # Number of steps within generation to visualize
        self.num_steps = num_steps           # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            for i in range(10):
                imgs_per_step = self.conditional_generate_imgs(pl_module, i)

                for j in range(imgs_per_step.shape[1]):
                    step_size = self.num_steps // self.vis_steps
                    imgs_to_plot = imgs_per_step[step_size-1::step_size,j]
                    grid = torchvision.utils.make_grid(imgs_to_plot.clamp_(min=-1.0, max=1.0), nrow=imgs_to_plot.shape[0], normalize=True)
                    trainer.logger.experiment.add_image("conditional_generation_{}".format(i), grid, global_step=trainer.current_epoch)

    def conditional_generate_imgs(self, pl_module, conditional_index):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = pl_module.generate_conditional_samples(start_imgs, steps=self.num_steps, step_size=1, conditional_index = conditional_index ,return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step

class SamplerCallback(pl.Callback):

    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs            
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            try:
                exmp_imgs = torch.stack(random.choices(pl_module.examples, k=self.num_imgs), dim=0)

                exmp_imgs = exmp_imgs.to(pl_module.device)
                grid = torchvision.utils.make_grid(exmp_imgs.clamp_(min=-1.0, max=1.0), nrow=4, normalize=True)
                trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)
            except Exception as e:
                print("Error in SamplerCallback: {}".format(e))
                pass

class OutlierCallback(pl.Callback):

    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs)[0].mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)

def main(args):

    pl.seed_everything(args.seed)
    transform_train = transforms.Compose([transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                lambda x: x + args.sigma * torch.rand_like(x)]
                )
    transform_test = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                #lambda x: x + args.sigma * torch.rand_like(x)]
                ])

    train_dataset = CIFAR10(root = DATASET_PATH, train = True, transform = transform_train, download = True)
    test_dataset = CIFAR10(root = DATASET_PATH, train = False, transform = transform_test, download = True)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, drop_last=True, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    trainer = pl.Trainer(default_root_dir=os.path.join(args.check_point_path, "CIFAR10"),
                         accelerator="gpu",
                         strategy="ddp_find_unused_parameters_true",
                         devices=args.devices,
                         max_epochs=args.max_epochs,
                         enable_progress_bar=True,
                         enable_model_summary=True,
                         enable_checkpointing=True,
                         callbacks=[ModelCheckpoint(dirpath="/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/notebooks/checkpoints/CIFAR10_test_wo_noise_sigma_0.1",filename="last", enable_version_counter=False,save_weights_only=False, every_n_train_steps=100, save_top_k=1, mode="min", monitor='loss_classify'),
                                    GenerateCallback(every_n_epochs=5),
                                    ConditionalGenerateCallback(every_n_epochs=5),
                                    SamplerCallback(every_n_epochs=5),
                                    OutlierCallback(),
                                    LearningRateMonitor("step")
                                   ])

    pretrained_filename = os.path.join(args.check_point_path, "CIFAR10.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        model = DeepEnergyModel(img_shape=(3,32,32),
                                batch_size=args.batch_size,
                                max_len=args.max_len,
                                alpha=args.alpha,
                                lr=args.lr,
                                beta1=args.beta1,
                                beta2=args.beta2,
                                weight_decay=args.weight_decay,
                                warmup_iters=args.warmup_iters,
                                decay_epoch=args.decay_epoch,
                                decay_rate=args.decay_rate,
                                seed=args.seed)
    if os.path.isfile("/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/notebooks/checkpoints/CIFAR10_final/ckpt/last.ckpt"):
        trainer.fit(model, train_loader, test_loader)
    else:
        trainer.fit(model, train_loader, test_loader)
    
    trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=6)
    parser.add_argument("--sigma",type=float,default=0.1)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--warmup_iters",type=int,default=1000)
    parser.add_argument("--max_len",type=int,default=10000)
    parser.add_argument("--alpha",type=float,default=0.01)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--beta1",type=float,default=0.9)
    parser.add_argument("--beta2",type=float,default=0.999)
    parser.add_argument("--weight_decay",type=float,default=0.0)
    parser.add_argument("--decay_epoch",type=int,default=50)
    parser.add_argument("--decay_rate",type=float,default=0.3)
    parser.add_argument("--max_epochs",type=int,default=150)
    parser.add_argument("--devices",type=int,default=4)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--drop_last",type=bool,default=True)
    parser.add_argument("--pin_memory",type=bool,default=False)
    parser.add_argument("--check_point_path",type=str,default="/work/home/maben/project/blue_whale_lab/projects/pareto_ebm/notebooks/checkpoints/CIFAR10_final")
    args = parser.parse_args()
    main(args)
