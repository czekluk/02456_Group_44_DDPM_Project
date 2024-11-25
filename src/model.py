import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import get_timestep_embedding, normalize, upsample, downsample

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, temb_ch, out_ch=None,  dropout=0.):
        """
        A Resnetblock that consists of two convolutional layers, t embeddings additions and a skip connection.
        For a visual description of the structure see docs/model.pdf
        Args:
            in_ch (int): Number of input channels.
            temb_ch (int): Number of channels for the time embedding.
            out_ch (int, optional): Number of output channels. 
            dropout (float, optional): Dropout rate. 
        """
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
        h = F.silu(x)
        h = self.conv1(h)
        # The t embeddign dimension is increased from (B, ch) to (B, ch, 1, 1) so that it can be added to the feature map (h).
        temb = F.silu(self.temb_proj(temb))[:, :, None, None]
        h += temb

        h = F.silu(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)
        
        if x.shape[1] != self.out_ch:
            x = self.shortcut_conv(x)
       

        return x + h

class AttnBlock(nn.Module):
    """
    Self attention is performed on each "pixel" of the input feature map. 
    "Which "pixels" have how big of an effect on which pixels?" 
    For a visual description of the structure see docs/model.pdf
    Attributes:
        norm (nn.GroupNorm): Group normalization layer.
        q (nn.Linear): Linear layer to compute query matrix.
        k (nn.Linear): Linear layer to compute key matrix.
        v (nn.Linear): Linear layer to compute value matrix.
        proj_out (nn.Linear): Linear layer for the final projection.
    Args:
        channels (int): Number of input channels.
    Methods:
        forward(x):
            Applies the attention mechanism to the input tensor x.
            Args:
                x (torch.Tensor): Input tensor of shape (B, C, H, W).
            Returns:
                torch.Tensor: Output tensor after applying attention, of shape (B, C, H, W).
    """
    
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
        # batch matrix multiplication: It just performs matrix multiplication for each batch seperately
        w = torch.bmm(q, k) * (C ** -0.5)
        w = torch.softmax(w, dim=-1)

        h = torch.bmm(w, v).reshape(B, H, W, C).permute(0, 3, 1, 2)  # B x C x H x W
        h = self.proj_out(h.reshape(B, H * W, C)).reshape(B, C, H, W)
        h= F.relu(h)
        
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels, out_chanels, with_conv):
        """
        It decreases the image size to (W//2,H//2), it can also change the number of chanels. 
        Usually out_chanels=in_chanels*2.
        Args:
            in_channels (int): Number of input channels.
            out_chanels (int): Number of output channels.
            with_conv (bool): If True, use a convolutional layer. If False, use an average pooling layer.
        """

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
        """
        Increase the image size to (W*2,H*2), it can also change th enumber of chanels. 
        Usually out_chanels=in_chanels//2.
        Args:
            in_channels (int): Number of input channels.
            out_chanels (int): Number of output channels.
            with_conv (bool): If True, use a convolutional layer. If False, use an upsampling layer.
        Attributes:
            with_conv (bool): Indicates whether a convolutional layer is used.
            conv (nn.Conv2d): Convolutional layer applied if with_conv is True.
            upsample (nn.Upsample): Upsampling layer applied if with_conv is False.
        """
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
    def __init__(self, ch_layer0, out_ch, ch_down_mult=(2, 4), num_res_blocks=2, chs_with_attention=[64], dropout=0., resamp_with_conv=True):
        """
        Initialize the UNet model.
        For a visual description of the structure see docs/model.pdf
        Args:
            - ch_layer0 (int): The number of channels in the initial layer. NOTE: This is not the number of input image chanels, those will be transformed to ch_layer0 channels. 
            out_ch (int): The number of output image channels.
            - ch_down_mult (list): It contains the multipliers for the number of channels in each layer. If ch_down_mult=(3, 6), we will have 3 layers (including the bottleneck), with [ch_layer0, ch_layer0*4, ch_layer0*6] channels.
            - num_res_blocks_per_layer (int): The number of residual blocks in a layer. 
            - attn_resolutions (list): Layers that have the number of chanels included in this list will have attention mechanisms. 
            If the list has [64, 256] and the our model has [64,128,256] chanels for each layer respectively, 
            then we will have attention mechanisms in the first and third layer.
            If you have 3 residual blocks in a layer, the layer will look as follows: [res,attn,res,attn,res,downsample(or upsample)]
            - dropout (float): Dropout rate. Default is 0.
            - resamp_with_conv (boo): Whether to use convolution for upsampling and downsampling. The alternative is to use nn.Upsample for upsampling and AveragePooling for dowssampling.
        """
        super().__init__()
        self.ch = ch_layer0
        self.out_ch = out_ch
        self.num_levels = len(ch_down_mult)
        ch_up_mult = [1] + list(ch_down_mult[:-1])
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = chs_with_attention
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv

        # Timestep embedding layers
        self.temb_dense0 = nn.Linear(ch_layer0, ch_layer0 * 4)
        self.temb_dense1 = nn.Linear(ch_layer0 * 4, ch_layer0 * 4)
        # Initial convolution
        self.conv_in = nn.LazyConv2d(ch_layer0, kernel_size=3, padding=1)

        # Downsampling block of Unet
        self.downs = nn.ModuleList()
        in_ch = ch_layer0
        for level in range(self.num_levels):

            for index in range(self.num_res_blocks):
                self.downs.append(ResNetBlock(in_ch, temb_ch=ch_layer0 * 4, out_ch=in_ch, dropout=dropout))
                if in_ch in chs_with_attention and index < self.num_res_blocks - 1:
                    self.downs.append(AttnBlock(in_ch))
            out_ch = ch_layer0 * ch_down_mult[level]
            self.downs.append(Downsample(in_ch, out_ch, self.resamp_with_conv))
            in_ch = out_ch

        # Bottleneck
        self.mid = nn.ModuleList()
        for index in range(self.num_res_blocks):
                self.mid.append(ResNetBlock(in_ch, temb_ch=ch_layer0 * 4, out_ch=in_ch, dropout=dropout))
                if in_ch in chs_with_attention and index < self.num_res_blocks - 1:
                    self.mid.append(AttnBlock(in_ch))


        # Upsampling block of the Unet
        self.ups = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            out_ch = ch_layer0 * ch_up_mult[level]
            self.ups.append(Upsample(in_ch, out_ch, self.resamp_with_conv))
            out_ch = out_ch*2 # concat with skip connection
            in_ch = out_ch
            for index in range(self.num_res_blocks + 1):
                self.ups.append(ResNetBlock(in_ch, temb_ch=ch_layer0 * 4, out_ch=in_ch, dropout=dropout))
                if out_ch//2 in chs_with_attention and index < self.num_res_blocks - 1: # so that it happens on the same level as in downsampling
                    self.ups.append(AttnBlock(out_ch))
                

        # Final normalization and output
        self.norm_out = nn.GroupNorm(num_groups=in_ch//2, num_channels=in_ch)
        self.conv_out = nn.Conv2d(in_ch, self.out_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        B, _, _, _ = x.shape
        assert t.dtype in [torch.int32, torch.int64]
        
        temb = get_timestep_embedding(t, self.temb_dense0.in_features)
        temb = F.silu(self.temb_dense0(temb))
        temb = F.silu(self.temb_dense1(temb))

        h = self.conv_in(x.float())
       
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

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)

class SimpleModel(Model):
    def __init__(self, ch_layer0, out_ch, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0., resamp_with_conv=True):
        """
        This class is meant to simplify the initialization of the Unet model by providing a more simple interface. 
        In turn, the Unet model cannot be initialized by arbitrary number of chanels for each layer. 
        The number of chanels always doubles as we go deeper into the model.
        For a visual description of the structure see docs/model.pdf
        Args:
            - ch_layer0 (int): The number of channels in the initial layer. 
            NOTE: This is not the number of input image chanels, those will be transformed to ch_layer0 channels. 
            - out_ch (int): The number of output image channels.
            - num_layers (int): The number of layers in the model. This includes the bottleneck.
            - num_res_blocks_per_layer (int): The number of residual blocks in a layer. 
            - layer_ids_with_attn (list): List of layer indices that will have attention mechanisms. If you have [0,2] in this list, you will have attention mechanisms in the first and third layer. 
            If you have 3 residual blocks in a layer, the layer will look as follows: [res,attn,res,attn,res,downsample(or upsample)]
            dropout (float): Dropout rate. Default is 0.
            - resamp_with_conv (bool): Whether to use convolution for upsampling and downsampling. The alternative is to use nn.Upsample for upsampling and AveragePooling for dowssampling.
        """
        ch_down_mult = []
        attn_resolutions = []
        if 0 in layer_ids_with_attn:
            attn_resolutions.append(ch_layer0)
        for i in range(1,num_layers):
            mult = 2**(i)
            ch = ch_layer0*(2**i)
            ch_down_mult.append(mult)
            if i in layer_ids_with_attn:
                attn_resolutions.append(ch)
        super().__init__(ch_layer0, out_ch, ch_down_mult, num_res_blocks_per_layer, attn_resolutions, dropout, resamp_with_conv)


# Load MNIST dataset
def test_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("DEVICE: ", device)
    
    # model = Model(ch=64, out_ch=1, ch_down_mult=(2, 4), num_res_blocks_per_layer=3, attn_resolutions=[64, 128], dropout=0.1, resamp_with_conv=True)
    model = SimpleModel(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
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