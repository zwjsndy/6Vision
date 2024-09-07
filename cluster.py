import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
class Encoder(nn.Module):
    """The encoder for VAE"""
    
    def __init__(self, image_size, input_dim, conv_dims, fc_dim, latent_dim):
        super().__init__()
        
        convs = []
        prev_dim = input_dim
        for conv_dim in conv_dims:
            convs.append(nn.Sequential(
                nn.Conv2d(prev_dim, conv_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ))
            prev_dim = conv_dim
        self.convs = nn.Sequential(*convs)
        
        prev_dim = (image_size // (2 ** len(conv_dims))) ** 2 * conv_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, fc_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(fc_dim, latent_dim)
        self.fc_log_var = nn.Linear(fc_dim, latent_dim)
                    
    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
class Decoder(nn.Module):
    """The decoder for VAE"""
    
    def __init__(self, latent_dim, image_size, conv_dims, output_dim):
        super().__init__()
        
        fc_dim = (image_size // (2 ** len(conv_dims))) ** 2 * conv_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU()
        )
        self.conv_size = image_size // (2 ** len(conv_dims))
        
        de_convs = []
        prev_dim = conv_dims[-1]
        for conv_dim in conv_dims[::-1]:
            de_convs.append(nn.Sequential(
                nn.ConvTranspose2d(prev_dim, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ))
            prev_dim = conv_dim
        self.de_convs = nn.Sequential(*de_convs)
        self.pred_layer = nn.Sequential(
            nn.Conv2d(prev_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.size(0), -1, self.conv_size, self.conv_size)
        x = self.de_convs(x)
        x = self.pred_layer(x)
        return x
class VAE(nn.Module):
    """VAE"""
    
    def __init__(self, image_size, input_dim, conv_dims, fc_dim, latent_dim):
        super().__init__()
        
        self.encoder = Encoder(image_size, input_dim, conv_dims, fc_dim, latent_dim)
        self.decoder = Decoder(latent_dim, image_size, conv_dims, input_dim)
        
    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var
    
    def compute_loss(self, x, recon, mu, log_var):
        """compute loss of VAE"""
        
        # KL loss
        kl_loss = (0.5*(log_var.exp() + mu ** 2 - 1 - log_var)).sum(1).mean()
        
        # recon loss
        recon_loss = F.binary_cross_entropy(recon, x, reduction="none").sum([1, 2, 3]).mean()
        
        return kl_loss + recon_loss
import ipaddress
def convert(seeds):
    result = []
    for line in seeds:
            line = line.split(":")
            for i in range(len(line)):
                if len(line[i]) == 4:
                    continue
                if len(line[i]) < 4 and len(line[i]) > 0:
                    zero = "0"*(4 - len(line[i]))
                    line[i] = zero + line[i]
                if len(line[i]) == 0:
                    zeros = "0000"*(9 - len(line))
                    line[i] = zeros
            result.append("".join(line)[:32])
    return result
def stdIPv6(addr: str):
    return ipaddress.ip_address(addr)
def str2ipv6(a: str):
    pattern = re.compile('.{4}')
    addr = ':'.join(pattern.findall(a))
    return str(stdIPv6(addr))
def hex2two(a):
    state_10 = int(a,16)
    str1= '{:04b}'.format(state_10)
    res=''
    res+='0'*(len(4*a)-len(str1))+str1
    return res
x = torch.rand(1, 1, 16, 16)

image_size = 16
conv_dims = [32, 64]
fc_dim = 128
latent_dim = 64

batch_size = 128
epochs = 50

transform=transforms.Compose([
    transforms.ToTensor()
])
import pickle
with open('./new_allseeds.pkl', 'rb') as f:
    seeds = pickle.load(f)
res=[]
for word in list(seeds.keys()):
    res+=seeds[word]
res=convert(res)
new_data=[]
for m,word in enumerate(res):
    temp1=[]
    word=hex2two(word)
    word=word+hex2two(res[m])
    for i in range(16):
        temp=[]
        for j in range(16):
            temp.append(float(word[i*16+j]))
        temp1.append(temp)
    new_data.append(temp1)
X_data = np.array(new_data)
X_data=np.expand_dims(X_data,axis=1)
X = torch.from_numpy(X_data).type(torch.FloatTensor)

from torch.utils.data import TensorDataset
dataset = TensorDataset(X,X)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

model = VAE(image_size, 1, conv_dims, fc_dim, latent_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print_freq = 200
for epoch in range(200):
    print("Start training epoch {}".format(epoch,))
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        recon, mu, log_var = model(images)
        loss = model.compute_loss(images, recon, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
            print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item()))
## visualize latent features

latent_zs = []
#targets = []
for i, (images, _) in enumerate(train_loader):
        images = images.cuda()
        with torch.no_grad():
            mu, log_var = model.encoder(images)
        latent_zs.append(mu.cpu().numpy())
        #targets.append(labels.numpy())
latent_zs = np.concatenate(latent_zs, 0)
#targets = np.concatenate(targets, 0)
from sklearn.cluster import KMeans
n_clusters=6
cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(latent_zs)
label= cluster.labels_
with open('./label.txt', 'w', encoding = 'utf-8') as f:
    for x in list(label):
        f.write(str(x) + '\n')
