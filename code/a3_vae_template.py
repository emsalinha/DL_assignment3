import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from datasets.bmnist import bmnist
from scipy.stats import norm

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()

        self.encode_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

        self.encode_std = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean_z, std_z = self.encode_mean(input), self.encode_std(input)

        return mean_z, std_z


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, output_dim=784):
        super().__init__()

        self.decode_mean = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean_x = self.decode_mean(input)

        return mean_x


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()

        self.stability_constant = 1e-8
        self.z_dim = z_dim
        self.input_dim = 784
        self.encoder = Encoder(hidden_dim, z_dim, input_dim)
        self.decoder = Decoder(hidden_dim, z_dim, input_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        input = input.view(-1, self.input_dim)

        mean_z, std_z = self.encoder(input)

        epsilon = torch.randn((1, self.z_dim))

        z = mean_z + std_z * epsilon

        mean_x = self.decoder(z)

        average_negative_elbo = self.loss(input, mean_x, mean_z, std_z)

        return average_negative_elbo

    def loss(self, input, mean_x, mean_z, std_z):

        x_hat = self.bernouilli(input, mean_x)
        loss_recon = -1 * torch.sum(x_hat, dim=1)

        loss_reg = self.KL_multivar(mean_z, std_z)

        negative_elbo = loss_recon + loss_reg
        average_negative_elbo = torch.mean(negative_elbo)

        return average_negative_elbo

    def bernouilli(self, input, mean_x):

        #assuming input values are either 0 or 1, adding the second part of the equation shouldnt make any difference

        mean_x = mean_x + self.stability_constant

        x_hat = input * torch.log(mean_x) + (1-input) * torch.log(1-mean_x)

        return x_hat

    def KL_multivar(self, mean, std):

        std = std + self.stability_constant
        var = std.pow(2)
        mean = mean.pow(2)

        KL = 0.5* torch.sum(var + mean - torch.log(std) - 1, dim=1)

        return KL

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        with torch.no_grad():
            z = torch.randn((n_samples, self.z_dim))
            samples = self.decoder(z)
            im_size = int(np.sqrt(self.input_dim))
            samples = samples.view(-1, 1, im_size, im_size)

        return samples


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    elbo_total = 0

    for i, batch in enumerate(data):
        elbo = model(batch)

        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()

        elbo_total += elbo.item()

    average_epoch_elbo = elbo_total / len(data)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def plot_sample(model, epoch, nrow, save=True):
    samples = model.sample(nrow*nrow)
    samples = make_grid(samples, nrow=nrow)
    samples = samples.numpy().transpose(1,2,0)
    file_name = 'samples_{}.jpg'.format(epoch)
    if save:
        plt.imsave(file_name, samples)
    else:
        plt.show()


def plot_manifold(model, nrow):
    # https://www.quora.com/How-can-I-draw-a-manifold-from-a-variational-autoencoder-in-Keras
    x_values = torch.linspace(.05, .95, nrow)
    y_values = torch.linspace(.05, .95, nrow)
    m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    canvas = np.empty((28 * nrow, 28 * nrow))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu =  m.icdf(torch.Tensor([xi, yi]))
            with torch.no_grad():
                x_mean = model.decoder(z_mu)
                canvas[(nrow - i - 1) * 28:(nrow - i) * 28, j * 28:(j + 1) * 28] = x_mean.numpy().reshape(28, 28)

    plt.imsave('manifold.jpg', canvas)

def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    prev_train_elbo = 0
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        plot_sample(model, epoch, 5, ARGS.save)
        if np.abs(prev_train_elbo - train_elbo) < 1:
            break
        else:
            prev_train_elbo = train_elbo

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    if ARGS.zdim == 2:
        plot_manifold(model, 15)

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--save', default=False, type=bool,
                        help='save or show plots')

    ARGS = parser.parse_args()

    main()
