import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

PROJECT_BASE_DIR =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_BASE_DIR, 'src')
sys.path.append(SRC_DIR)
from utils import get_timestep_embedding, normalize, upsample, downsample
from unet import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ModelClassifierFreeGuidance(Model):
    def __init__(self, ch_layer0, out_ch, ch_down_mult=(2, 4), num_res_blocks=2, chs_with_attention=[64], dropout=0., resamp_with_conv=True, lambda_cf=1000.):
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
        super().__init__(ch_layer0, out_ch, ch_down_mult, num_res_blocks, chs_with_attention, dropout, resamp_with_conv)
        self.label_embedding = nn.Embedding(10, ch_layer0 * 4)
        self.p_guidance = lambda_cf / (lambda_cf + 1)
    
    def forward(self, x, t, label):
        B, _, _, _ = x.shape
        assert t.dtype in [torch.int32, torch.int64]

        temb = get_timestep_embedding(t, self.temb_dense0.in_features)
        temb = F.silu(self.temb_dense0(temb))
        temb = F.silu(self.temb_dense1(temb))
        if torch.rand(1).item() < self.p_guidance:
            temb += F.silu(self.label_embedding(label))
        return self._forward(x, temb)
    
class SimpleModelClassFreeGuidance(ModelClassifierFreeGuidance):
    def __init__(self, ch_layer0, out_ch, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0., resamp_with_conv=True, lambda_cf=1.):
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
        super().__init__(ch_layer0, out_ch, ch_down_mult, num_res_blocks_per_layer, attn_resolutions, dropout, resamp_with_conv, lambda_cf)