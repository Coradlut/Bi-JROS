import torch
import torch.nn as nn
from torch.autograd import Variable
from layers import SpatialTransformer, ResizeTransform, VecInt, conv_block, predict_flow, MatchCost,\
    DoubleConv, Down, Up, Up3, OutConv, conv3D
from torch.distributions.normal import Normal
from torch import sub, add, cat

class Feature_Extractor(nn.Module):
    def __init__(self, n_channels):
        super(Feature_Extractor, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return x

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()

        self.up1 = Up3(512, 128, tag=2)
        self.up2 = Up3(128, 64, tag=3)
        self.up3 = Up3(64, 32, tag=3)

    def forward(self, encoded_src, encoded_tgt):
        s1, s2, s3, s4 = encoded_src
        t1, t2, t3, t4 = encoded_tgt
        x4 = torch.cat([s4, t4], dim=1)
        x = self.up1(x4, s3, t3)
        x = self.up2(x, s2, t2)
        x = self.up3(x, s1, t1)
        return x

class Reg_Decoder(nn.Module):
    def __init__(self, inshape):
        super(Reg_Decoder, self).__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.decoder = Decoder2()
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(32, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure transformer
        self.transformer = SpatialTransformer()

    def forward(self, source, encoded_src, encoded_tgt):
        x = self.decoder(encoded_src, encoded_tgt)
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow

class Seg_Decoder(nn.Module):
    def __init__(self, inshape, n_classes):
        super(Seg_Decoder, self).__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.decoder = Decoder()

        self.n_classes = n_classes
        self.outc = OutConv(32, n_classes)

    def forward(self, encoded_src):
        x1, x2, x3, x4 = encoded_src
        x = self.decoder(x1, x2, x3, x4)

        logits = self.outc(x)
        
        return logits

class Architecture(nn.Module):
    def __init__(self, inshape, n_channels=1, n_classes=14):
        super(Architecture, self).__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.fea_extractor = Feature_Extractor(n_channels)
        self.Reg_Decoder = Reg_Decoder(inshape)
        self.Seg_Decoder = Seg_Decoder(inshape, n_classes)

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        encoded_src = self.fea_extractor(source)
        encoded_tgt = self.fea_extractor(target)

        y_source, pos_flow = self.Reg_Decoder(source, encoded_src, encoded_tgt)
        
        # return y_source, pos_flow
        s_logits = self.Seg_Decoder(encoded_src)
        t_logits = self.Seg_Decoder(encoded_tgt)
        
        # return y_source, pos_flow, s_logits, t_logits, encoded_tgt
        return y_source, pos_flow, s_logits, t_logits
        # return y_source, pos_flow, t_logits


