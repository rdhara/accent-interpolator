from preprocess import generate_data_loaders
from matplotlib import pyplot as plt

import torch
import seaborn as sns
from torch import nn, optim
from torch.nn import functional as F

cuda = torch.cuda.is_available()
torch.manual_seed(42)
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


class VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=20):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, hidden_dim)
        self.fc22 = nn.Linear(400, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x.view(-1, 128), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train(epoch, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 2000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == '__main__':
    NUM_EPOCHS = 100
    train_trajectory, test_trajectory = [], []
    train_loader_dr, test_loader_dr = generate_data_loaders('DR7', batch_size=4)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_trajectory.append(train(epoch, train_loader_dr))
        test_trajectory.append(test(test_loader_dr))


# Plotting code
# Single Plot
# plt.figure()
# plt.title('New England VAE')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (REC + KL)')
# plt.plot(train_trajectory, c='b', label='Training')
# plt.plot(test_trajectory, c='r', label='Validation')
# plt.legend()
# plt.savefig('dr_1.png', dpi=300)
# # Subplots
# sns.set_style('whitegrid')
# sns.set_context('paper', rc={'axes.labelsize': 8})
# sns.set_palette('muted', color_codes=True)
#
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(6,4))
# plt.suptitle('Individual Accent VAEs')
# axs = [ax1, ax2, ax3, ax4]
# titles = ['New England', 'North Midland', 'Southern', 'Western']
#
# for i in range(4):
#     train, test = trajs[i]
#     ax = axs[i]
#     ax.set_title(titles[i])
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('$\mathcal{L}$ = REC + KL')
#     ax.set_xlim([0, 100])
#     ax.set_ylim([0, 500])
#     ax.set_yticks([0, 100, 200, 300, 400, 500])
#     ax.plot(train, c='b', label='Training', lw=2)
#     ax.plot(test, c='r', label='Validation', lw=2)
#     ax.legend()
#
# f.tight_layout()
# plt.subplots_adjust(top=0.85)
#
# plt.savefig('accent_vaes.png', dpi=300)

