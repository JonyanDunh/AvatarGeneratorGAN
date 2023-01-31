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


images = torch.load(r"/mnt/d/data/saved_tensor/gan_images/ganyu-images_128.pt").permute(3, 0, 1, 2).to(
    dtype=torch.float32, device="cuda") / 255


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
            View(3 * 18 * 18),
            nn.Linear(3 * 18 * 18, 1),
            # nn.Sigmoid()
        )
        self.model128 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 3, kernel_size=8, stride=2), nn.LeakyReLU(0.2),
            # View(3 * 10 * 10),
            # nn.Linear(3 * 10 * 10, 1),
            # nn.Sigmoid()
        )

        self.fc1 = nn.Linear(3 * 10 * 10, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=0.0001)
        self.loss_fn = nn.BCEWithLogitsLoss().cuda()
        self.loss_train = 0.0
        self.items = 0
        self.counter = 0

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

        self.loss_train += loss.item()
        self.items += 1
        self.counter += 1

        writer.add_scalar('Discriminator Training loss/times', loss.item(), self.counter)


class Generator_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model512 = nn.Sequential(
            nn.Linear(100, 3 * 19 * 19),
            nn.LeakyReLU(0.2),
            View((1, 3, 19, 19)),
            nn.ConvTranspose2d(3, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=16, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 3, kernel_size=16, stride=2, padding=1), nn.BatchNorm2d(3), nn.Sigmoid(),

            View((1, 3, 512, 512))
        )
        self.model128 = nn.Sequential(
            nn.ConvTranspose2d(3, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 256, kernel_size=8, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 3, kernel_size=8, stride=2, padding=1), nn.BatchNorm2d(3), nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(100, 3 * 11 * 11)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.loss_train = 0.0
        self.items = 0
        self.counter = 0
        self.Epoch=0

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
        writer.add_scalar('Generator Training loss/times', loss.item(), self.counter)
        if self.items == 0:
            # plt.imshow(
            #     ((Generator_outputs[0].detach()) * 255).reshape(3, 128, 128).permute(1, 2, 0).to(dtype=torch.int).cpu())
            # plt.show()
            # writer.add_image('Images-Epoch-{}'.format(self.Epoch), Generator_outputs[0])
            writer.add_image('Images', Generator_outputs[0])
            self.Epoch+=1
        self.items += 1
        self.counter += 1


t = time.time()


def training_loop():
    Discriminator = Discriminator_Model().cuda()
    Discriminator = torch.compile(Discriminator)
    Generator = Generator_Model().cuda()
    Generator = torch.compile(Generator)
    for Epoch in range(100000):
        for imgs in torch.utils.data.DataLoader(images, batch_size=1, shuffle=False):
            Discriminator.trains(imgs, torch.ones(imgs.shape[0], 1).cuda())
            Discriminator.trains(Generator(torch.randn(imgs.shape[0], 100).cuda()).detach(),
                                 torch.zeros(imgs.shape[0], 1).cuda())
            Generator.trains(Discriminator, torch.randn(imgs.shape[0], 100).cuda(), torch.ones(imgs.shape[0], 1).cuda())
        print('{} Epoch {}, Discriminator Training loss {}, Generator Training loss {} '.format(
            datetime.datetime.now(), Epoch,
            Discriminator.loss_train / Discriminator.items, Generator.loss_train / Generator.items))
        writer.add_scalar('Discriminator Training loss/Epoch', Discriminator.loss_train / Discriminator.items, Epoch)
        writer.add_scalar('Generator Training loss/Epoch', Generator.loss_train / Generator.items, Epoch)
        Discriminator.clear()
        Generator.clear()
        for i, (name, param) in enumerate(Discriminator.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram("Discriminator", param, Epoch)
        for i, (name, param) in enumerate(Generator.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram("Generator", param, Epoch)
        if Epoch % 10 == 0:
            global t
            torch.save(Generator.state_dict(), '/mnt/d/data/saved_model/gan_images/gan_images-{}.pt'.format(int(t)))


# def validate(model):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for imags, labels in torch.utils.data.DataLoader(
#                 [(fake_images, torch.tensor([1], dtype=torch.float32, device="cuda")) for fake_images in
#                  images], batch_size=1024, shuffle=False):
#             outputs = model(imags)
#             print(outputs)
#             total += imags.shape[0]
#             correct += float(outputs.sum())
#     print("Accuracy : {:.4f}".format(correct / total))


# validate(distinguish_model)

training_loop()
