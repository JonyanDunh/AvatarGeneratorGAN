import torch
from pathlib import Path
from torchvision import transforms
import torchvision
from torchvision.io import image as Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import datetime
import math
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

torch.set_float32_matmul_precision('high')
torch.set_default_tensor_type(torch.FloatTensor)
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()


def image2tensor():
    p = transforms.Compose([transforms.Resize((128, 128))])
    torch.save(torch.stack(
        [Image.read_image(str(dir)) for dir in
         [e for e in Path(r'D:\data/gan_images/ganyu/ganyu-final/').iterdir()]],
        dim=3), r"D:\data/saved_tensor/gan_images/ganyu-images_512.pt")


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


class Discriminator_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model512 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 3, kernel_size=16, stride=2), nn.LeakyReLU(0.2),
        )
        self.model128 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 3, kernel_size=8, stride=2), nn.LeakyReLU(0.2),
        )

        self.fc1 = nn.Linear(3 * 10 * 10, 1)
        self.fc2 = nn.Linear(3 * 18 * 18, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=0.0003)
        self.loss_fn = nn.BCEWithLogitsLoss().cuda()

    def clear(self):
        self.loss_train = 0.0
        self.items = 0

    def forward(self, inputs):  # 直接运行模型
        out = self.model128(inputs)
        out = out.view(-1, 3 * 10 * 10)
        out = self.fc1(out)
        # print(out.shape)
        return out

    def trains(self, inputs, targets):
        outputs = self.forward(inputs)
        with torch.cuda.amp.autocast():
            loss = self.loss_fn(outputs, targets)
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        return loss.item()


class Generator_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model512 = nn.Sequential(
            nn.ConvTranspose2d(3, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 3, kernel_size=16, stride=2, padding=1), nn.BatchNorm2d(3), nn.Sigmoid(),

        )
        self.model128 = nn.Sequential(
            nn.ConvTranspose2d(3, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 3, kernel_size=8, stride=2, padding=1), nn.BatchNorm2d(3), nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(100, 3 * 11 * 11)
        self.fc2 = nn.Linear(100, 3 * 19 * 19)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.08)
        self.loss_train = 0.0
        self.items = 0
        self.counter = 0
        self.Epoch = 0

    def clear(self):
        self.loss_train = 0.0
        self.items = 0

    def forward(self, inputs):
        out = self.fc1(inputs)
        out = F.leaky_relu(out)
        out = out.view(-1, 3, 11, 11)
        out = self.model128(out)
        return out

    def trains(self, Discriminator, inputs, targets):
        Generator_outputs = self.forward(inputs)
        Discriminator_outputs = Discriminator.forward(Generator_outputs)
        with torch.cuda.amp.autocast():
            loss = Discriminator.loss_fn(Discriminator_outputs, targets)

        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.loss_train += loss.item()

        if self.items == 0:
            writer.add_image('Image/Epoch', Generator_outputs[0],self.Epoch)
            self.Epoch += 1
        if self.counter % 500 == 0:
            # plt.imshow(
            #     ((Generator_outputs[0].detach()) * 255).reshape(3, 128, 128).permute(1, 2, 0).to(dtype=torch.int).cpu())
            # plt.show()
            writer.add_image('Image/Times', Generator_outputs[0],self.counter)
        self.items += 1
        self.counter += 1
        return loss.item()


t = time.time()

# Generator = Generator_Model().cuda()
# Generator = torch.compile(Generator)
# Generator.load_state_dict(torch.load('gan_images-Generator-128-1675214311-Epoch-1600.pt'))
# images=Generator(torch.randn(100, 100).cuda())
# plt.figure(figsize=(15, 15))
# for i in range(100):
#     plt.subplot(10, 10, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(
#         ((images[i].detach()) * 255).reshape(3, 128, 128).permute(1, 2, 0).to(dtype=torch.int).cpu(), cmap=plt.cm.binary)
# plt.show()

images = torch.load(r"/mnt/d/data/saved_tensor/gan_images/ganyu-images_128.pt").permute(3, 0, 1, 2).to(
    dtype=torch.float32, device="cuda") / 255


def training_loop():
    Discriminator = Discriminator_Model().cuda()
    Discriminator = torch.compile(Discriminator)
    Generator = Generator_Model().cuda()
    Generator = torch.compile(Generator)
    Generator.load_state_dict(torch.load('gan_images-Generator-128-1675214311-Epoch-1600.pt'))
    counter = 0

    for Epoch in range(100000):
        Discriminator_real_loss_gross = 0.0
        Discriminator_fake_loss_gross = 0.0
        Generator_loss_gross = 0.0
        counter_in_epoch = 0
        for imgs in torch.utils.data.DataLoader(images, batch_size=1, shuffle=False):
            Discriminator_real_loss = Discriminator.trains(imgs, torch.ones(imgs.shape[0], 1).cuda())
            Discriminator_fake_loss = Discriminator.trains(Generator(torch.rand(imgs.shape[0], 100).cuda()).detach(),
                                                           torch.zeros(imgs.shape[0], 1).cuda())
            Generator_loss = Generator.trains(Discriminator, torch.randn(imgs.shape[0], 100).cuda(),
                                              torch.ones(imgs.shape[0], 1).cuda())
            Discriminator_real_loss_gross += Discriminator_real_loss
            Discriminator_fake_loss_gross += Discriminator_fake_loss
            Generator_loss_gross += Generator_loss
            writer.add_scalars('Training loss/Times',
                               {'Discriminator Real with Fake': (Discriminator_real_loss + Discriminator_fake_loss) / 2,
                                'Discriminator Real': Discriminator_real_loss,
                                'Discriminator Fake': Discriminator_fake_loss,
                                'Generator': Generator_loss
                                }, counter)
            writer.add_scalars('Discriminator Training loss/Times',
                               {'Real with Fake': (Discriminator_real_loss + Discriminator_fake_loss) / 2,
                                'Real': Discriminator_real_loss,
                                'Fake': Discriminator_fake_loss
                                }, counter)
            writer.add_scalar('Generator Training loss/Times', Generator_loss, counter)
            counter += 1
            counter_in_epoch += 1
        print('{} Epoch {}, Training loss:\n{}'.format(
            datetime.datetime.now(), Epoch,
            {'Discriminator Real with Fake': (
                                                     Discriminator_real_loss_gross + Discriminator_fake_loss_gross) / (2 * counter_in_epoch),
             'Discriminator Real': Discriminator_real_loss_gross / counter_in_epoch,
             'Discriminator Fake': Discriminator_fake_loss_gross / counter_in_epoch,
             'Generator': Generator_loss_gross / counter_in_epoch}))
        writer.add_scalars('Training loss/Epoch',
                           {'Discriminator Real with Fake': (
                                                                    Discriminator_real_loss_gross + Discriminator_fake_loss_gross) / (2 * counter_in_epoch),
                            'Discriminator Real': Discriminator_real_loss_gross / counter_in_epoch,
                            'Discriminator Fake': Discriminator_fake_loss_gross / counter_in_epoch,
                            'Generator': Generator_loss_gross / counter_in_epoch}, Epoch)
        writer.add_scalars('Discriminator Training loss/Epoch', {
            'Real with Fake': (Discriminator_real_loss_gross + Discriminator_fake_loss_gross) / (2 * counter_in_epoch),
            'Real': Discriminator_real_loss_gross / counter_in_epoch,
            'Fake': Discriminator_fake_loss_gross / counter_in_epoch}, Epoch)
        writer.add_scalar('Generator Training loss/Epoch', Generator_loss_gross / counter_in_epoch, Epoch)
        Discriminator.clear()
        Generator.clear()
        if Epoch % 100 == 0:
            global t
            torch.save(Discriminator.state_dict(),
                       '/mnt/d/data/saved_model/gan_images/gan_images-Discriminator-{}-{}-Epoch-{}.pt'.format(
                           images.shape[2], int(t), Epoch))
            torch.save(Generator.state_dict(),
                       '/mnt/d/data/saved_model/gan_images/gan_images-Generator-{}-{}-Epoch-{}.pt'.format(
                           images.shape[2], int(t), Epoch))


training_loop()
