import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config: dict):
        super(Generator, self).__init__()
        number_channel = config["number_channel"]
        image_size = config["image_size"] 
        d_hidden = config["d_hidden"]
        num_classes = config["num_classes"]

        self.deconv_latent = nn.Sequential(
            nn.ConvTranspose2d(d_hidden, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
        )

        self.deconv_classes = nn.Sequential(
            nn.ConvTranspose2d(num_classes, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(image_size * 16, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            nn.ConvTranspose2d( image_size, number_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y):
        # x shape: [batch_size, d_hidden]
        # y shape: [batch_size, num_classes]
        # output shape: [batch_size, number_channel, image_size, image_size]

        x1 = self.deconv_latent(x.view(x.size(0), x.size(1), 1, 1))
        y1 = self.deconv_classes(y.view(y.size(0), y.size(1), 1, 1))
        xy = torch.cat([x1,y1], dim = 1)
        return self.main(xy)



class Discriminator(nn.Module):
    def __init__(self, config: dict):
        super(Discriminator, self).__init__()
        number_channel = config["number_channel"]
        image_size = config["image_size"] 
        d_hidden = config["d_hidden"]
        num_classes = config["num_classes"]

        self.conv_img = nn.Sequential(
            nn.Conv2d(number_channel, image_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_classes = nn.Sequential(
            nn.Conv2d(num_classes, image_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            nn.Conv2d(image_size * 2, image_size * 2, 4, 2, 1),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 8, d_hidden, 4, 1, 0)
        )
        self.fc = nn.Sequential(
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        # x shape: [batch_size, number_channel, image_size, image_size]
        # y shape: [batch_size, num_classes, image_size, image_size]
        # output shape: [batch_size, 1]

        x1 = self.conv_img(x)
        y1 = self.conv_classes(y)
        xy = torch.cat([x1,y1], dim = 1)
        return self.fc(self.main(xy).squeeze(-1).squeeze(-1))