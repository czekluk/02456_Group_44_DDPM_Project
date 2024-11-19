import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# Assuming the `nn` module has helper functions like conv2d, dense, etc.

def nonlinearity(x):
    return F.silu(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    device = timesteps.device  # Get the device of `timesteps`

    assert len(timesteps.shape) == 1  # Ensure `timesteps` is a 1D tensor

    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    
    # Ensure that all tensors are created directly on `device`
    emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
    emb = timesteps[:, None].float().to(device) * emb[None, :]  # Multiply on `device`
    
    # Concatenate sinusoidal and cosinusoidal embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).to(device)
    
    # Pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1)).to(device)
    
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def normalize(x, temb, name):
    # Replacing group_norm from tf.contrib
    num_groups = 32  # You may change this as needed
    return nn.GroupNorm(num_groups=num_groups, num_channels=x.shape[1])(x)

def upsample(x, with_conv):
    B, C, H, W = x.shape
    x = nn.Upsample(size=(H * 2, W * 2), mode='nearest')(x)
    if with_conv:
        x = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, stride=1, padding=1)(x)
    return x

def downsample(x, with_conv):
    B, C, H, W = x.shape
    if with_conv:
        x = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, stride=2, padding=1)(x)
    else:
        x = F.avg_pool2d(x, 2, stride=2)
    return x

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, temb_ch, out_ch=None,  dropout=0.):
        super().__init__()
        self.out_ch = out_ch or in_ch
        self.dropout = dropout

        self.norm1 = nn.GroupNorm(num_groups=in_ch//2, num_channels=in_ch, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=self.out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, self.out_ch)

        self.norm2 = nn.GroupNorm(num_groups=in_ch//2, num_channels=self.out_ch)
        self.conv2 = nn.Conv2d(in_channels=self.out_ch, out_channels=self.out_ch, kernel_size=3, padding=1)

        
        self.shortcut_conv = nn.Conv2d(in_channels=in_ch, out_channels=self.out_ch, kernel_size=1)

    def forward(self, x, temb):
        x=self.norm1(x)
        h = nonlinearity(x)
        h = self.conv1(h)
        
        temb = nonlinearity(self.temb_proj(temb))[:, :, None, None]
        h += temb

        h = nonlinearity(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)

        if x.shape[1] != self.out_ch:
            x = self.shortcut_conv(x)
       

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=channels//2, num_channels=channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).permute(0, 2, 3, 1).reshape(B, H * W, C)  # B x (HW) x C
        q = self.q(h)
        k = self.k(h).permute(0, 2, 1)  # B x C x (HW)
        v = self.v(h)

        w = torch.bmm(q, k) * (C ** -0.5)
        w = torch.softmax(w, dim=-1)

        h = torch.bmm(w, v).reshape(B, H, W, C).permute(0, 3, 1, 2)  # B x C x H x W
        h = self.proj_out(h.reshape(B, H * W, C)).reshape(B, C, H, W)
        
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels, out_chanels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_chanels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        else:
            return self.pool(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_chanels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_chanels, kernel_size=3, stride=1, padding=1)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        if self.with_conv:
            return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))
        else:
            return self.upsample(x)

class Model(nn.Module):
    def __init__(self, ch, out_ch, ch_down_mult=(2, 4, 8, 16), num_res_blocks=2,
                 attn_resolutions=[64], dropout=0., resamp_with_conv=True):
        super().__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.num_levels = len(ch_down_mult)
        ch_up_mult = [1] + list(ch_down_mult[:-1])
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv

        # Timestep embedding layers
        self.temb_dense0 = nn.Linear(ch, ch * 4)
        self.temb_dense1 = nn.Linear(ch * 4, ch * 4)
        # Initial convolution
        self.conv_in = nn.LazyConv2d(ch, kernel_size=3, padding=1)

        # Downsampling block of Unet
        self.downs = nn.ModuleList()
        in_ch = ch
        for level in range(self.num_levels):

            for index in range(self.num_res_blocks):
                self.downs.append(ResNetBlock(in_ch, temb_ch=ch * 4, out_ch=in_ch, dropout=dropout))
                if in_ch in attn_resolutions and index < self.num_res_blocks - 1:
                    self.downs.append(AttnBlock(in_ch))
            out_ch = ch * ch_down_mult[level]
            self.downs.append(Downsample(in_ch, out_ch, self.resamp_with_conv))
            in_ch = out_ch

        # Bottleneck
        self.mid = nn.ModuleList([
            ResNetBlock(in_ch, temb_ch=ch * 4, out_ch=in_ch, dropout=dropout),
            AttnBlock(in_ch),
            ResNetBlock(in_ch, temb_ch=ch * 4, out_ch=in_ch, dropout=dropout)
        ])

        # Upsampling block of the Unet
        self.ups = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            out_ch = ch * ch_up_mult[level]
            self.ups.append(Upsample(in_ch, out_ch, self.resamp_with_conv))
            out_ch = out_ch*2 # concat with skip connection
            in_ch = out_ch
            for index in range(self.num_res_blocks + 1):
                self.ups.append(ResNetBlock(in_ch, temb_ch=ch * 4, out_ch=in_ch, dropout=dropout))
                if out_ch//2 in attn_resolutions and index < self.num_res_blocks - 1: # so that it happens on the same level as in downsampling
                    self.ups.append(AttnBlock(out_ch))
                

        # Final normalization and output
        self.norm_out = nn.GroupNorm(num_groups=in_ch//2, num_channels=in_ch)
        self.conv_out = nn.Conv2d(in_ch, self.out_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        B, _, _, _ = x.shape
        assert t.dtype in [torch.int32, torch.int64]
        
        temb = get_timestep_embedding(t, self.temb_dense0.in_features)
        temb = nonlinearity(self.temb_dense0(temb))
        temb = nonlinearity(self.temb_dense1(temb))

        h = self.conv_in(x)
       
        hs = []
        for layer in self.downs:
            if isinstance(layer, ResNetBlock): #start of layer
                h = layer(h, temb)
            elif isinstance(layer, Downsample): #end of layer
                hs.append(h)
                h = layer(h)
            else:
                h = layer(h)

        for layer in self.mid:
            if isinstance(layer, ResNetBlock): 
                h = layer(h, temb)
            else:
                h = layer(h)

        for layer in self.ups:
            if isinstance(layer, ResNetBlock):
                h = layer(h, temb)
            elif isinstance(layer, Upsample): #start of layer
                h = layer(h)
                from_down = hs.pop()
                assert from_down.shape == h.shape
                h=torch.cat([h,from_down ], dim=1)
            else:
                h = layer(h)

        h = nonlinearity(self.norm_out(h))
        return self.conv_out(h)

# Load MNIST dataset
def test_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("DEVICE: ", device)
    # Initialize the model
    model = Model(ch=64, out_ch=1, ch_down_mult=(2, 4), num_res_blocks=2, attn_resolutions=[64], dropout=0.1, resamp_with_conv=True)
    model = model.to(device)

    # Test the forward process
    for i, (images, labels) in enumerate(train_loader):
        if i >= 2:
            break
        images = images.to(device)
        t = torch.randint(0, 1000, (images.size(0),)).to(device)
        output = model(images, t)
        print(output.shape)
        import matplotlib.pyplot as plt

        # Save the output image plot
        output_image = output[0].detach().cpu().numpy().squeeze()
        plt.imshow(output_image, cmap='gray')
        plt.axis('off')
        plt.savefig('output_image.png')
        plt.close()

        # Define a simple loss function
        criterion = nn.MSELoss()

        # Generate a random noisy image
        noisy_image = torch.randn_like(images).to(device)

        # Forward pass
        output = model(noisy_image, t)

        # Compute loss
        loss = criterion(output, images)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Print the loss
        print(f'Loss: {loss.item()}')

if __name__ == "__main__":
    test_model()