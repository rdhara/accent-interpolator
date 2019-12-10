from itertools import chain
from utils import generate_data_loaders
from matplotlib import pyplot as plt

import torch
import seaborn as sns
from torch import nn, optim
from torch.nn import functional as F


cuda = torch.cuda.is_available()
torch.manual_seed(42)
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader_dr1, test_loader_dr1 = generate_data_loaders('DR1', batch_size=4)
train_loader_dr5, test_loader_dr5 = generate_data_loaders('DR5', batch_size=45)

input_dim = 128
hidden_dim = 32
NUM_EPOCHS = 250


class VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=20):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc3 = nn.Linear(hidden_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x, encoder):
        mu, logvar = encoder(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class SharedEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=20):
        super(SharedEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, hidden_dim)
        self.fc22 = nn.Linear(400, hidden_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)


model_1 = VAE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
model_2 = VAE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
encoder = SharedEncoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

optimizer = optim.Adam(
    params=chain(*[
        model_1.parameters(),
        model_2.parameters(),
        encoder.parameters()
    ]),
    lr=1e-3
)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x.view(-1, 128), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train(epoch, train_loader_1, train_loader_2):

    model_1.train()
    model_2.train()
    encoder.train()

    train_loss = 0

    for batch_idx, data in enumerate(train_loader_1):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model_1(data, encoder)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    for batch_idx, data in enumerate(train_loader_2):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model_2(data, encoder)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader_1.dataset) + len(train_loader_2.dataset)
    print('====> Epoch: {}\n\tTraining set loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


def test(test_loader_1, test_loader_2):
    model_1.eval()
    model_2.eval()
    encoder.eval()

    test_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader_1):
            data = data.to(device)
            recon_batch, mu, logvar = model_1(data, encoder)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

        for i, data in enumerate(test_loader_2):
            data = data.to(device)
            recon_batch, mu, logvar = model_2(data, encoder)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader_1.dataset) + len(test_loader_2.dataset)
    print('\tTest set loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == '__main__':
    train_trajectory, test_trajectory = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train(epoch, train_loader_dr1, train_loader_dr5)
        train_trajectory.append(tr_loss)
        vl_loss = test(train_loader_dr1, train_loader_dr5)
        test_trajectory.append(vl_loss)

# # Plotting Code
# sns.set_style('whitegrid')
# sns.set_context('paper', rc={'axes.labelsize': 14, 'axes.titlesize': 14})

# sns.set_palette('muted', color_codes=True)
# plt.figure(figsize=(6,4))
# plt.xlabel('Epoch')
# plt.ylabel('$\mathcal{L}_{tot}$')
# plt.xlim([0,100])
# plt.ylim([0, 500])
# plt.title('Jointly Training New England and Southern VAEs')
# plt.plot(train_trajectory, c='b', label='Training', lw=2)
# plt.plot(test_trajectory, c='r', label='Validation', lw=2)
# plt.legend();
# plt.tight_layout()
# plt.savefig('dr_1_5_join_traning.png', dpi=300)

