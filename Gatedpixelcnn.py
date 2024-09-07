import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
class MaskedConv2d(nn.Conv2d):
    """
    Implements a conv2d with mask applied on its weights.
    
    Args:
        mask (torch.Tensor): the mask tensor.
        in_channels (int) – Number of channels in the input image.
        out_channels (int) – Number of channels produced by the convolution.
        kernel_size (int or tuple) – Size of the convolving kernel
    """
    
    def __init__(self, mask, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('mask', mask[None, None])
        
    def forward(self, x):
        self.weight.data *= self.mask # mask weights
        return super().forward(x)
    

class VerticalStackConv(MaskedConv2d):

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height (k//2, k), but for simplicity, we stick with masking here.
        self.mask_type = mask_type
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        mask = torch.zeros(kernel_size)
        mask[:kernel_size[0]//2, :] = 1.0
        if self.mask_type == "B":
            mask[kernel_size[0]//2, :] = 1.0

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)
        

class HorizontalStackConv(MaskedConv2d):

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        self.mask_type = mask_type
        
        if isinstance(kernel_size, int):
            kernel_size = (1, kernel_size)
        assert kernel_size[0] == 1
        if "padding" in kwargs:
            if isinstance(kwargs["padding"], int):
                kwargs["padding"] = (0, kwargs["padding"])
        
        mask = torch.zeros(kernel_size)
        mask[:, :kernel_size[1]//2] = 1.0
        if self.mask_type == "B":
            mask[:, kernel_size[1]//2] = 1.0

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)
# visualize the receptive field for VerticalStackConv and HorizontalStackConv
# we can compute the gradients of the input to imshow the receptive field

inp_img = torch.zeros(1, 1, 11, 11)
inp_img.requires_grad_()

def show_center_recep_field(img, out):
    """
    Calculates the gradients of the input with respect to the output center pixel,
    and visualizes the overall receptive field.
    Inputs:
        img - Input image for which we want to calculate the receptive field on.
        out - Output features/loss which is used for backpropagation, and should be
              the output of the network/computation graph.
    """
    # Determine gradients, the center pixel
    loss = out[0, :, img.shape[2]//2, img.shape[3]//2].sum() # L1 loss for simplicity
    # Retain graph as we want to stack multiple layers and show the receptive field of all of them
    loss.backward(retain_graph=True)
    img_grads = img.grad.abs()
    img.grad.fill_(0) # Reset grads

    # Plot receptive field
    img = img_grads.squeeze().cpu().numpy()
    fig, ax = plt.subplots(1, 2)
    pos = ax[0].imshow(img)
    ax[1].imshow(img > 0)
    # Mark the center pixel in red if it doesn't have any gradients,
    # which is the case for standard autoregressive models)
    show_center = (img[img.shape[0]//2, img.shape[1]//2] == 0)
    if show_center:
        center_pixel = np.zeros(img.shape + (4,))
        center_pixel[center_pixel.shape[0]//2, center_pixel.shape[1]//2, :] = np.array([1.0, 0.0, 0.0, 1.0])
    for i in range(2):
        ax[i].axis('off')
        if show_center:
            ax[i].imshow(center_pixel)
    ax[0].set_title("Weighted receptive field")
    ax[1].set_title("Binary receptive field")
    plt.show()
    plt.close()

# we don't use conv, so the receptive field is only the center pixel
show_center_recep_field(inp_img, inp_img)
# we first visualize the original masked_conv
kernel_size = 3
mask_A = torch.zeros((3, 3))
mask_A[:kernel_size//2, :] = 1.0
mask_A[kernel_size//2, :kernel_size//2] = 1.0

masked_conv = MaskedConv2d(mask_A, 1, 1, 3, padding=1)
masked_conv.weight.data.fill_(1)
masked_conv.bias.data.fill_(0)
masked_conv_img = masked_conv(inp_img)
show_center_recep_field(inp_img, masked_conv_img)
# use mask_type B
mask_B = mask_A.clone()
mask_B[kernel_size//2, kernel_size//2] = 1.
masked_conv = MaskedConv2d(mask_B, 1, 1, 3, padding=1)
masked_conv.weight.data.fill_(1)
masked_conv.bias.data.fill_(0)

for l_idx in range(4):
    masked_conv_img = masked_conv(masked_conv_img)
    print(f"Layer {l_idx+2}")
    show_center_recep_field(inp_img, masked_conv_img)
    
# there is a “blind spot” on the right upper side
# visualize HorizontalStackConv
horiz_conv = HorizontalStackConv("A", 1, 1, 3, padding=1)
horiz_conv.weight.data.fill_(1)
horiz_conv.bias.data.fill_(0)
horiz_img = horiz_conv(inp_img)
show_center_recep_field(inp_img, horiz_img)
# visualize VerticalStackConv
vert_conv = VerticalStackConv("A", 1, 1, 3, padding=1)
vert_conv.weight.data.fill_(1)
vert_conv.bias.data.fill_(0)
vert_img = vert_conv(inp_img)
show_center_recep_field(inp_img, vert_img)
# combine the two by adding, which is what we expect
horiz_img = vert_img + horiz_img
show_center_recep_field(inp_img, horiz_img)
# Initialize convolutions with equal weight to all input pixels
horiz_conv = HorizontalStackConv("B", 1, 1, 3, padding=1)
horiz_conv.weight.data.fill_(1)
horiz_conv.bias.data.fill_(0)
vert_conv = VerticalStackConv("B", 1, 1, 3, padding=1)
vert_conv.weight.data.fill_(1)
vert_conv.bias.data.fill_(0)

# note we use mask_type A for the first layer, but after first layer we should use mask_type B

# We reuse our convolutions for the 4 layers here. Note that in a standard network,
# we don't do that, and instead learn 4 separate convolution. As this cell is only for
# visualization purposes, we reuse the convolutions for all layers.
for l_idx in range(4):
    vert_img = vert_conv(vert_img)
    horiz_img = horiz_conv(horiz_img) + vert_img
    print(f"Layer {l_idx+2}")
    show_center_recep_field(inp_img, horiz_img)
class GatedMaskedConv(nn.Module):

    def __init__(self, in_channels, kernel_size=3, dilation=1):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        self.conv_vert = VerticalStackConv("B", in_channels, 2*in_channels, kernel_size, padding=padding,
                                          dilation=dilation)
        self.conv_horiz = HorizontalStackConv("B", in_channels, 2*in_channels, kernel_size, padding=padding,
                                             dilation=dilation)
        self.conv_vert_to_horiz = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        self.conv_horiz_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
# GatedPixelCNN

class GatedPixelCNN(nn.Module):
    
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        
        # Initial first conv with mask_type A
        self.conv_vstack = VerticalStackConv("A", in_channels, channels, 3, padding=1)
        self.conv_hstack = HorizontalStackConv("A", in_channels, channels, 3, padding=1)
        # Convolution block of PixelCNN. use dilation instead of 
        # downscaling used in the encoder-decoder architecture in PixelCNN++
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(channels),
            GatedMaskedConv(channels, dilation=2),
            GatedMaskedConv(channels),
            GatedMaskedConv(channels, dilation=4),
            GatedMaskedConv(channels),
            GatedMaskedConv(channels, dilation=2),
            GatedMaskedConv(channels)
        ])
        
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # first convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))
        return out
# visualize GaGatedPixelCNN
test_model = GatedPixelCNN(1, 64, 1)
inp = torch.zeros(1, 1, 16, 16)
inp.requires_grad_()
out = test_model(inp)
show_center_recep_field(inp, out.squeeze(dim=2))
del inp, out, test_model
import ipaddress
import re
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
image_size = 16
in_channels = 1
out_channels = 1
channels = 128 # hidden channels

batch_size = 64
epochs =15

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
label=[]
with open('./label.txt', 'r') as f:
    for line in f:
        label.append(line.strip('\n'))
f.close()
all_data=[]
for lb in list(set(label)):
    temp=[]
    for i,word in enumerate(label):
        if word==lb:
            temp.append(res[i])
    all_data.append(temp)
new_data=[]
for cl in range(6):
    all_data1=all_data[cl]  
    for m,word in enumerate(all_data1):
        temp1=[]
        word=hex2two(word)
        word=word+hex2two(all_data1[m])
        for i in range(16):
            temp=[]
            for j in range(16):
                temp.append(float(word[i*16+j]))
            temp1.append(temp)
        new_data.append(temp1)
    for m,word in enumerate(all_data1[:-1]):
        temp1=[]
        word=hex2two(word)
        word=word+hex2two(all_data1[m+1])
        for i in range(16):
            temp=[]
            for j in range(16):
                temp.append(float(word[i*16+j]))
            temp1.append(temp)
        new_data.append(temp1)
    for m,word in enumerate(all_data1[:-2]):
        temp1=[]
        word=hex2two(word)
        word=word+hex2two(all_data1[m+2])
        for i in range(16):
            temp=[]
            for j in range(16):
                temp.append(float(word[i*16+j]))
            temp1.append(temp)
        new_data.append(temp1)
    for m,word in enumerate(all_data1[:-3]):
        temp1=[]
        word=hex2two(word)
        word=word+hex2two(all_data1[m+3])
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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = GatedPixelCNN(in_channels, channels, out_channels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    print_freq = 1000
    epochs=40
    for epoch in range(epochs):
        print("Start training epoch {}".format(epoch,))
        for i, (images, labels) in enumerate(train_loader):
            images = (images > 0.33).float() # convert to 0, 1
            images = images.cuda()
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item())) 
    PATH='./model/'+str(cl)+'.pth'
    torch.save(model, PATH)