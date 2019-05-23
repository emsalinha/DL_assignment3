import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.generate = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )


    def forward(self, z):

        img = self.generate(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminate = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, img):
        # return discriminator score for img

        pred = self.discriminate(img)

        return pred


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            #imgs.cuda()

            batch_size = imgs.shape[0]
            imgs = imgs.view(batch_size, 784)

            # Train Generator
            # -------------------
            z = torch.randn(batch_size, args.latent_dim)
            fake_imgs = generator(z)
            pred_fake = discriminator(fake_imgs)
            loss_generator = -torch.mean(torch.log(pred_fake))

            optimizer_G.zero_grad()
            loss_generator.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            z = torch.randn(batch_size, args.latent_dim)
            fake_imgs = generator(z).detach()
            pred_real = discriminator(imgs)
            pred_fake = discriminator(fake_imgs)
            loss_discriminator = -torch.mean(torch.log(pred_real) + torch.log(1 - pred_fake))

            optimizer_D.zero_grad()
            loss_discriminator.backward()
            optimizer_D.step()


            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(fake_imgs[:25].view(-1, 1, 28, 28),
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

        print(f"[Epoch {epoch}] gen loss: {loss_generator} disc loss: {loss_discriminator}")


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                           ])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
