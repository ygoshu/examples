from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import utils, datasets, transforms
from torchvision.utils import save_image
import sys
sys.path.insert(1, '/home/yag3/WGAN/pytorch-wgan')
from utils.fashion_mnist import MNIST

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='N',
                    help='latent dim (default: 20)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

trans = transforms.Compose([
    transforms.Scale(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/train_mnist', train=True, download=True,
                   transform=trans),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_emnist_dataset = MNIST(root='./data/test_emnist', train=False, download=True, transform=trans, few_shot_class=5, test_emnist=True) 
test_emnist_loader  = torch.utils.data.DataLoader(test_emnist_dataset ,  batch_size=args.test_batch_size, shuffle=True)

test_mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/test_mnist', train=False, download=True, transform=trans),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, args.latent_dim)
        self.fc3 = nn.Linear(args.latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        torch.save(model.state_dict(), './vae.pkl')
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, test_loader):
    model.eval()
    model.load_state_dict(torch.load('vae.pkl'))
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'vae_results/reconstruction_' + str(epoch) + '_'+ str(is_emnist) + '.png', nrow=n)
                break
            
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def optimizeZ(test_loader):
       if not os.path.exists('gen_vae_res/'):
           os.makedirs('gen_vae_res/')
       model.load_state_dict(torch.load('vae.pkl'))
       z = torch.randn(args.test_batch_size, args.latent_dim).to(device)
       #z = Variable(Tensor(np.random.normal(0, 1, (args.test_batch_size, args.latent_dim))))
       z.requires_grad = True
       print("Checking if z requires Gradient")
       print(z.requires_grad)
       learning_rate = 0.002
       opt_iter = 0

       loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
       print("self.epochs")
       epochs = 12000
       images = None
       for i, (image, _) in enumerate(test_loader):
           if i != 4733:
               continue
           images = Variable(image.to(device).type(Tensor))
           grid_org = utils.make_grid(images)
           utils.save_image(grid_org, 'gen_vae_res/orig_ae_res_{}_{}.png'.format( str(opt_iter).zfill(3), str(is_emnist)))
           break
       optimizer = torch.optim.Adam([z], lr=learning_rate)
       grid = None
       og = None
       for epoch in range(epochs):
             #generate image
             x_recon = model.decode(z)

             if opt_iter == 0:
                 og_recon = x_recon.to(device)
             #calculate reconstruction loss 
             loss = loss_fn(x_recon, images.view(args.test_batch_size,-1))  #test_loader andk get first img)

             #zero out gradient so that the previous calculated gradient doesn't add on to the current calculated grad
             optimizer.zero_grad()

             #calculate gradient
             loss.backward()

             #update scale 
             optimizer.step()
             
             if opt_iter % 1000 == 0:
                 print("Iter {}, loss {}".format(str(opt_iter), str(loss.item())))
                 x_recon = x_recon.view(args.test_batch_size, 1, 28, 28)
                 utils.save_image(x_recon, 'gen_vae_res/gen_ae_res_{}_{}.png'.format(str(is_emnist), str(opt_iter).zfill(3)), nrow=x_recon.size(0))

             opt_iter += 1

       comparison = torch.cat([og_recon.view(args.test_batch_size, 1, 28, 28), 
                                     images, 
                                     x_recon.view(args.test_batch_size, 1, 28, 28)])
       utils.save_image(comparison.cpu() ,
                        'gen_vae_res/comparison_{}.png'.format(is_emnist),
                          nrow=args.test_batch_size )


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
#        train(epoch)
        is_emnist = 1
#        test(epoch, test_emnist_loader)
        optimizeZ(test_emnist_loader)
        is_emnist = 0
#        test(epoch, test_mnist_loader)
        optimizeZ(test_mnist_loader)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'vae_results/sample_' + str(epoch) + '.png')
