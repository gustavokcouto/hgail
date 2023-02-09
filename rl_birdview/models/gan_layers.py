# modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py

import datetime
import os
import time
import sys

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

from rl_birdview.models.discriminator import ExpertDataset


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, pad=None):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.pad = pad

    def forward(self, x, skip_input=None):
        x = self.model(x)
        if not self.pad is None:
            x = torch.nn.functional.pad(x, self.pad)
        if not skip_input is None:
            x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256, dropout=0.5)
        self.flatten = nn.Flatten()

        self.up1 = UNetUp(384, 256, dropout=0.5)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64, pad=(1, 0, 1, 0))
        self.up4 = UNetUp(128, 128)

        self.final = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
            nn.Tanh(),
        )

        self.linear = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.up_linear1 = UNetUp(64, 128)
        self.up_linear2 = UNetUp(128, 128)

    def forward(self, x, cmd, traj):
        # U-Net generator with skip connections from encoder to decoder
        linear_input = torch.cat((cmd, traj), 1)
        linear_features = self.linear(linear_input)
        l1 = linear_features.unsqueeze(dim=2).unsqueeze(dim=2)
        l2 = self.up_linear1(l1)
        l3 = self.up_linear2(l2)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u0 = torch.cat([d4, l3], dim=1)
        u1 = self.up1(u0, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3)

        final = self.final(u4)
        return final

##############################
#        Discriminator
##############################


class GanDiscriminator(nn.Module):
    def __init__(self):
        super(GanDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_images = nn.Sequential(
            *discriminator_block(12, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256)
        )

        self.linear = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.up_linear1 = UNetUp(64, 128)
        self.up_linear2 = UNetUp(128, 128)

        self.model_all = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(384, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_birdview, img_rgb, cmd, traj):
        # Concatenate image and condition image by channels to produce input
        images = torch.cat([img_rgb, img_birdview], dim=1)
        images_feats = self.model_images(images)
        linear_input = torch.cat((cmd, traj), 1)
        linear_features = self.linear(linear_input)
        l1 = linear_features.unsqueeze(dim=2).unsqueeze(dim=2)
        l2 = self.up_linear1(l1)
        l3 = self.up_linear2(l2)

        feats_all = torch.cat([images_feats, l3], dim=1)
        feats_all = self.model_all(feats_all)
        return feats_all


class GanFakeBirdview(nn.Module):
    def __init__(self, batches_done=0):
        super(GanFakeBirdview, self).__init__()

        self.generator = GeneratorUNet()
        self.generator = self.generator.cuda()
        # generator_variables = torch.load('saved_models/fake_birdview/generator_8.pth', map_location='cuda')
        # self.generator.load_state_dict(generator_variables)

        self.discriminator = GanDiscriminator()
        self.discriminator = self.discriminator.cuda()
        # discriminator_variables = torch.load('saved_models/facades/discriminator_48.pth', map_location='cuda')
        # self.discriminator.load_state_dict(discriminator_variables)

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_GAN = self.criterion_GAN.cuda()
        self.criterion_pixelwise = torch.nn.L1Loss()
        self.criterion_pixelwise = self.criterion_pixelwise.cuda()
        self.lambda_pixel = 100
        self.patch = (1, 10, 10)
        self.sample_interval = 50
        self.checkpoint_interval = 2
        self.batches_done = batches_done
        self.transforms = transforms.Compose([
            transforms.Resize((192, 192)),
        ])

        os.makedirs("images/fake_birdview", exist_ok=True)
        os.makedirs("images/birdview", exist_ok=True)
        os.makedirs("saved_models/fake_birdview", exist_ok=True)

        self.val_dataloader = torch.utils.data.DataLoader(
            ExpertDataset(
                'gail_experts',
                n_routes=1,
                n_eps=1,
            ),
            batch_size=10,
            shuffle=True,
            num_workers=1,
        )

    def train_batch(self, obs_dict, num_timesteps=0):
        # Model inputs
        rgb_array = []
        for rgb_key in ['central_rgb', 'left_rgb', 'right_rgb']:
            rgb_img = (obs_dict[rgb_key].float() - 127.5) / 127.5
            rgb_img = self.transforms(rgb_img)
            rgb_array.append(rgb_img)

        rgb = torch.cat(rgb_array, dim=1)
        real_rgb = torch.autograd.Variable(rgb.type(torch.cuda.FloatTensor))
        real_cmd = torch.autograd.Variable(obs_dict["cmd"].type(torch.cuda.FloatTensor))
        real_traj = torch.autograd.Variable(obs_dict["traj"].type(torch.cuda.FloatTensor))
        birdview = (obs_dict["birdview"].float() - 127.5) / 127.5
        real_birdview = torch.autograd.Variable(birdview.type(torch.cuda.FloatTensor))

        # Adversarial ground truths
        valid = torch.autograd.Variable(torch.cuda.FloatTensor(np.ones((real_rgb.size(0), *self.patch))), requires_grad=False)
        fake = torch.autograd.Variable(torch.cuda.FloatTensor(np.zeros((real_rgb.size(0), *self.patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        self.optimizer_G.zero_grad()

        # GAN loss
        fake_birdview = self.generator(real_rgb, real_cmd, real_traj)
        pred_fake = self.discriminator(fake_birdview, real_rgb, real_cmd, real_traj)
        loss_GAN = self.criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = self.criterion_pixelwise(fake_birdview, real_birdview)

        # Total loss
        loss_G = loss_GAN + self.lambda_pixel * loss_pixel

        loss_G.backward()

        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Real loss
        pred_real = self.discriminator(real_birdview, real_rgb, real_cmd, real_traj)
        loss_real = self.criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = self.discriminator(fake_birdview.detach(), real_rgb, real_cmd, real_traj)
        loss_fake = self.criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        self.optimizer_D.step()

        # If at sample interval save image
        if self.batches_done % self.sample_interval == 0:
            with torch.no_grad():
                self.sample_images(num_timesteps)

        self.batches_done += 1

        return loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item()

    def val_batch(self, obs_dict):
        with torch.no_grad():
            # Model inputs
            rgb_array = []
            for rgb_key in ['central_rgb', 'left_rgb', 'right_rgb']:
                rgb_img = (obs_dict[rgb_key].float() - 127.5) / 127.5
                rgb_img = self.transforms(rgb_img)
                rgb_array.append(rgb_img)

            rgb = torch.cat(rgb_array, dim=1)
            real_rgb = torch.autograd.Variable(rgb.type(torch.cuda.FloatTensor))
            real_cmd = torch.autograd.Variable(obs_dict["cmd"].type(torch.cuda.FloatTensor))
            real_traj = torch.autograd.Variable(obs_dict["traj"].type(torch.cuda.FloatTensor))
            birdview = (obs_dict["birdview"].float() - 127.5) / 127.5
            real_birdview = torch.autograd.Variable(birdview.type(torch.cuda.FloatTensor))

            # Adversarial ground truths
            valid = torch.autograd.Variable(torch.cuda.FloatTensor(np.ones((real_rgb.size(0), *self.patch))), requires_grad=False)
            fake = torch.autograd.Variable(torch.cuda.FloatTensor(np.zeros((real_rgb.size(0), *self.patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            # GAN loss
            fake_birdview = self.generator(real_rgb, real_cmd, real_traj)
            pred_fake = self.discriminator(fake_birdview, real_rgb, real_cmd, real_traj)
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(fake_birdview, real_birdview)

            # Total loss
            loss_G = loss_GAN + self.lambda_pixel * loss_pixel

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Real loss
            pred_real = self.discriminator(real_birdview, real_rgb, real_cmd, real_traj)
            loss_real = self.criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = self.discriminator(fake_birdview.detach(), real_rgb, real_cmd, real_traj)
            loss_fake = self.criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

        return loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item()

    def pretrain(self):
        dataloader = torch.utils.data.DataLoader(
            ExpertDataset(
                'gail_experts',
                n_routes=10,
                n_eps=1,
            ),
            batch_size=48,
            shuffle=True,
            num_workers=8,
        )

        n_epochs = 10
        prev_time = time.time()
        for epoch in range(n_epochs):
            for i, batch in enumerate(dataloader):
                obs_dict, _ = batch
                gan_disc_loss, gan_generator_loss, gan_pixel_loss, gan_loss = self.train_batch(obs_dict)
                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                    % (
                        epoch,
                        n_epochs,
                        i,
                        len(dataloader),
                        gan_disc_loss,
                        gan_generator_loss,
                        gan_pixel_loss,
                        gan_loss,
                        time_left,
                    )
                )

            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), "saved_models/fake_birdview/generator_%d.pth" % epoch)
                torch.save(self.discriminator.state_dict(), "saved_models/fake_birdview/discriminator_%d.pth" % epoch)

    def sample_images(self, num_timesteps):
        """Saves a generated sample from the validation set"""
        batch = next(iter(self.val_dataloader))
        obs_dict, _ = batch
        # Model inputs
        rgb_array = []
        for rgb_key in ['central_rgb', 'left_rgb', 'right_rgb']:
            rgb_img = (obs_dict[rgb_key].float() - 127.5) / 127.5
            rgb_img = self.transforms(rgb_img)
            rgb_array.append(rgb_img)

        rgb = torch.cat(rgb_array, dim=1)
        real_rgb = torch.autograd.Variable(rgb.type(torch.cuda.FloatTensor))
        real_cmd = torch.autograd.Variable(obs_dict["cmd"].type(torch.cuda.FloatTensor))
        real_traj = torch.autograd.Variable(obs_dict["traj"].type(torch.cuda.FloatTensor))
        real_birdview = torch.autograd.Variable(obs_dict["birdview"].type(torch.cuda.FloatTensor))
        fake_birdview = self.generator(real_rgb, real_cmd, real_traj)
        fake_birdview = (fake_birdview * 127.5) + 127.5
        for i_img in range(fake_birdview.shape[0]):
            save_image(fake_birdview[i_img], "images/fake_birdview/{}_{}_{}.png".format(self.batches_done, num_timesteps, i_img))
            save_image(real_birdview[i_img], "images/birdview/{}_{}_{}.png".format(self.batches_done, num_timesteps, i_img))

    def fill_expert_dataset(self, dataloader):
        fake_birdview_list = [None for _ in range(len(dataloader.dataset))]
        batch_size = 32

        for batch in dataloader:
            obs_dict_batch, _ = batch
            start_slice_idx = 0
            while start_slice_idx < obs_dict_batch['birdview'].shape[0]:
                end_slice_idx = min(start_slice_idx+batch_size, obs_dict_batch['birdview'].shape[0])
                obs_dict = dict([(k, v[start_slice_idx:end_slice_idx]) for k, v in obs_dict_batch.items()])
                with torch.no_grad():
                    rgb_array = []
                    for rgb_key in ['central_rgb', 'left_rgb', 'right_rgb']:
                        rgb_img = (obs_dict[rgb_key].float() - 127.5) / 127.5
                        rgb_img = self.transforms(rgb_img)
                        rgb_array.append(rgb_img)

                    rgb = torch.cat(rgb_array, dim=1)
                    real_rgb = torch.autograd.Variable(rgb.type(torch.cuda.FloatTensor))
                    real_cmd = torch.autograd.Variable(obs_dict["cmd"].type(torch.cuda.FloatTensor))
                    real_traj = torch.autograd.Variable(obs_dict["traj"].type(torch.cuda.FloatTensor))

                    # GAN loss
                    fake_birdview = self.generator(real_rgb, real_cmd, real_traj)
                    fake_birdview = (fake_birdview * 127.5) + 127.5
                    fake_birdview = fake_birdview.cpu()

                item_idx_array = obs_dict['item_idx'].cpu().numpy()
                for item_idx in range(fake_birdview.shape[0]):
                    fake_birdview_list[item_idx_array[item_idx]] = fake_birdview[item_idx]
                start_slice_idx += batch_size
        fake_birdview_tensor = torch.stack(fake_birdview_list)
        return fake_birdview_tensor