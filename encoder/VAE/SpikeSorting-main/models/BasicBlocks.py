# T. S. Liang @ HKU EEE, Jan. 29th 2026
# Email: sliang57@connect.hku.hk
# Basic blocks for residual autoencoder.

from typing import Optional

import torch.nn as nn
import torch
import torch.nn.functional as F

def channel_average(x, factor=2):
    """
    Input:  [B, C_in]  (e.g., [B, 32])
    Output: [B, C_out] (e.g., [B, 16]) where C_out = C_in // scale_factor
    """
    batch_size, channels = x.size()

    if channels % factor != 0:
        raise ValueError(f"Input channels ({channels}) must be divisible by scale_factor ({factor})")
    
    out_channels = channels // factor

    x = x.view(batch_size, out_channels, factor)
    
    x = x.mean(dim=2)
    
    return x

def channel_duplicate(x, factor=2):
    """
    Input:  [B, C_in]  (e.g., [B, 16])
    Output: [B, C_out] (e.g., [B, 32]) where C_out = C_in * scale_factor
    """
    batch_size, channels = x.size()
    
    x = x.unsqueeze(2)
    
    x = x.expand(batch_size, channels, factor)
    
    x = x.reshape(batch_size, channels * factor)
    
    return x
    
class MLPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act = True
    ):
        super(MLPLayer, self).__init__()

        self.mlp = nn.Linear(
            in_channels,
            out_channels
        )

        self.norm = nn.BatchNorm1d(out_channels) if act else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.mlp(x)
        x = self.norm(x)

        x = self.act(x)

        return x

class ChannelAvg(nn.Module):

    def __init__(
        self,
        in_channels,
        factor = 2
    ):

        super().__init__()
        
        self.factor = factor
        out_channels = in_channels

        self.mlp = nn.Linear(
            in_channels,
            out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.mlp(x)
        
        x = channel_average(x, factor = self.factor)

        return x

class ChannelDup(nn.Module):

    def __init__(
        self,
        in_channels,
        factor = 2
    ):

        super().__init__()
        
        self.factor = factor
        out_channels = in_channels

        self.mlp = nn.Linear(
            in_channels,
            out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.mlp(x)
        
        x = channel_duplicate(x, factor = self.factor)

        return x

#################################################################################
#                             Functional Blocks                                 #
#################################################################################

class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
    ):
        super(ResidualBlock, self).__init__()

        self.main = main
        self.shortcut = shortcut

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:

        return self.main(x)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
        return res

#################################################################################
#                             Enc/Dec Functions                                 #
#################################################################################
class ChannelAvgMLPBlock(ResidualBlock):

    # this class only implement the channel averaging and does not down sample.
    
    def __init__(
        self,
        in_channels: int,
    ):

        out_channels = in_channels // 2

        main_branch = MLPLayer(in_channels = in_channels, out_channels = out_channels)
        shortcut_branch = ChannelAvg(in_channels  = in_channels)

        super().__init__(
            main=main_branch, 
            shortcut=shortcut_branch
        )

class ChannelDupMLPBlock(ResidualBlock):

    # this class only implement the channel averaging and does not down sample.
    
    def __init__(
        self,
        in_channels: int,
    ):

        out_channels = in_channels * 2

        main_branch = MLPLayer(in_channels = in_channels, out_channels = out_channels)
        shortcut_branch = ChannelDup(in_channels = in_channels)

        super().__init__(
            main=main_branch, 
            shortcut=shortcut_branch
        )

class ResMLPBlock(ResidualBlock):

    # this class only implement the channel averaging and does not down sample.
    
    def __init__(
        self,
        in_channels: int,
        act = True
    ):

        out_channels = in_channels

        main_branch = MLPLayer(in_channels = in_channels, out_channels = out_channels, act = act)
        shortcut_branch = nn.Identity()

        super().__init__(
            main=main_branch, 
            shortcut=shortcut_branch
        )

#################################################################################
#                             Enc/Dec Layers.                                   #
#################################################################################

class Encoder(nn.Module):

    def __init__(self, waveform_len = 32, num_channels = 20):

        super().__init__()

        self.waveform_proj_block = ResMLPBlock(in_channels = waveform_len)
        self.channel_max_min_proj_block = MLPLayer(in_channels = num_channels, out_channels = waveform_len)

        self.ds_mlp1 = ChannelAvgMLPBlock(in_channels = waveform_len*2)
        self.ds_mlp2 = ChannelAvgMLPBlock(in_channels = waveform_len)
        self.ds_mlp3 = ChannelAvgMLPBlock(in_channels = waveform_len//2)

    def forward(self, waveform, channel_max_min):

        """

            inputs: 
            waveform: [B, 32], the time-step dependent waveform from the most salient channel.
            channel_max_min: [B, 20], the amplitude of all of the channels, with the minimum and maximum values.

            output: [B, 8]: latent vector.
        """

        wf = self.waveform_proj_block(waveform) # [B, 32]
        am_emb = self.channel_max_min_proj_block(channel_max_min)  # [B, 32]

        x = torch.concat([wf, am_emb], dim = 1) # [B, 64]

        x = self.ds_mlp1(x) # [B, 32]
        x = self.ds_mlp2(x) # [B, 16]
        x = self.ds_mlp3(x) # [B, 8]

        return x

class Decoder(nn.Module):

    def __init__(self, waveform_len = 32, num_channels = 20):
        super(Decoder, self).__init__()

        # 1. Upsampling Stream (Reverse of Encoder's Downsampling)
        # Encoder: 64 -> 32 -> 16 -> 8
        # Decoder: 8 -> 16 -> 32 -> 64
        
        # input [B, 8] -> [B, 16]
        self.us_mlp3 = ChannelDupMLPBlock(in_channels = waveform_len // 4) 
        
        # input [B, 16] -> [B, 32]
        self.us_mlp2 = ChannelDupMLPBlock(in_channels = waveform_len // 2)
        
        # input [B, 32] -> [B, 64]
        self.us_mlp1 = ChannelDupMLPBlock(in_channels = waveform_len)

        # 2. Reconstruction Heads (Symmetric to Encoder's Input Heads)
        
        # Reconstruct Waveform: [B, 32] -> [B, 32]
        # Symmetric to Encoder's waveform_proj_block
        self.waveform_recon_block = ResMLPBlock(in_channels = waveform_len, act = False)
        
        # Reconstruct Channel Info: [B, 32] -> [B, 20]
        # Reverse of Encoder's channel_max_min_proj_block
        self.channel_recon_block = MLPLayer(
            in_channels = waveform_len, 
            out_channels = num_channels,
            act = False
        )

    def forward(self, x: torch.Tensor):
        """
        input: 
            x: [B, 8] latent vector
        
        outputs:
            wf_recon: [B, 32] reconstructed waveform
            am_recon: [B, 20] reconstructed channel info
        """

        # 1. Upsampling
        x = self.us_mlp3(x) # [B, 16]
        x = self.us_mlp2(x) # [B, 32]
        x = self.us_mlp1(x) # [B, 64]

        # 2. Splitting (Reverse of torch.concat)
        # Split [B, 64] back into two [B, 32] tensors
        wf_feat, am_feat = torch.chunk(x, chunks=2, dim=1)

        # 3. Reconstruction
        wf_recon = self.waveform_recon_block(wf_feat)   # [B, 32]
        am_recon = self.channel_recon_block(am_feat)    # [B, 32] -> [B, 20]

        return wf_recon, am_recon

if __name__ == "__main__":

    x = torch.rand([1, 32])
    channel_max_min = torch.rand([1, 20])

    model = Encoder(waveform_len = 32, num_channels = 20)

    x = model(x, channel_max_min)
    
    print(x.shape)