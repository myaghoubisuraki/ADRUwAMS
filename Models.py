import torch
import torch.nn as nn

############################  Model Part

# Multiscale Spatial attention 
class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism. This mechanism uses a convolutional layer followed by a sigmoid function
    to compute an attention map. This map is used to weigh the input features, emphasizing important 
    spatial locations while suppressing less useful ones.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv3x3 = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv3d(in_channels, 1, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv3d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map3x3 = self.sigmoid(self.conv3x3(x))
        attn_map5x5 = self.sigmoid(self.conv5x5(x))
        attn_map7x7 = self.sigmoid(self.conv7x7(x))

        attn_map = attn_map3x3 + attn_map5x5 + attn_map7x7
        return x * attn_map

class AttentionGate(nn.Module):
    """
    Attention Gate mechanism. This is a more complex attention mechanism that works by generating 
    two feature transformations of the input and the gating signal. The attention coefficients are 
    computed as a sigmoid activation of the sum of these transformations.
    """
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(AttentionGate, self).__init__()

        if inter_channels is None:
            inter_channels = in_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv3d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=inter_channels, num_channels=inter_channels),
            nn.ReLU(inplace=True)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=inter_channels, num_channels=inter_channels),
            nn.ReLU(inplace=True)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

class ResBlock(nn.Module):
    """
    Residual block. This is a block commonly used in ResNet-like architectures. It consists of two 
    convolutional layers each followed by a group normalization and a ReLU activation. There is a shortcut 
    connection that bypasses the two convolutions which helps in dealing with the vanishing gradient problem.
    """
    def __init__(self, in_channels, out_channels, num_groups=8, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class UNet3D(nn.Module):
    """
    3D U-Net architecture. This architecture is a variation of the original U-Net and it's designed to work with 
    3D inputs. It consists of an encoder (downsampling path), a bottleneck, and a decoder (upsampling path).
    In this variant, residual blocks are used in both the encoder and decoder paths. There are also attention gates 
    that are used to weigh the features in the upsampling path with respect to those in the downsampling path.
    """
    def __init__(self, in_channels=4, out_channels=3, init_features=8,):
        super(UNet3D, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            ResBlock(in_channels, init_features),
            ResBlock(init_features, init_features)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            ResBlock(init_features, init_features * 2),
            ResBlock(init_features * 2, init_features * 2)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = nn.Sequential(
            ResBlock(init_features * 2, init_features * 4),
            ResBlock(init_features * 4, init_features * 4)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2,)
        self.encoder4 = nn.Sequential(
            ResBlock(init_features * 4, init_features * 8),
            ResBlock(init_features * 8, init_features * 8)
        )
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(init_features * 8, init_features * 16),
            ResBlock(init_features * 16, init_features * 16)
        )

        # Attention Gates
        self.att4 = AttentionGate(init_features * 8, init_features * 8)
        self.att3 = AttentionGate(init_features * 4, init_features * 4)
        self.att2 = AttentionGate(init_features * 2, init_features * 2)
        self.att1 = AttentionGate(init_features, init_features)

        # Spatial Attention
        self.spatial_attention4 = SpatialAttention(init_features * 16)
        self.spatial_attention3 = SpatialAttention(init_features * 8)
        self.spatial_attention2 = SpatialAttention(init_features * 4)
        self.spatial_attention1 = SpatialAttention(init_features * 2)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            ResBlock(init_features * 16, init_features * 8),
            ResBlock(init_features * 8, init_features * 8)
        )
        self.upconv3 = nn.ConvTranspose3d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            ResBlock(init_features * 8, init_features * 4),
            ResBlock(init_features * 4, init_features * 4)
        )
        self.upconv2 = nn.ConvTranspose3d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            ResBlock(init_features * 4, init_features * 2),
            ResBlock(init_features * 2, init_features * 2)
        )
        self.upconv1 = nn.ConvTranspose3d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            ResBlock(init_features * 2, init_features),
            ResBlock(init_features, init_features)
        )
        
        # Classifier
        self.cls = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.att4(enc4, dec4)), dim=1)
        dec4 = self.spatial_attention4(dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.att3(enc3, dec3)), dim=1)
        dec3 = self.spatial_attention3(dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.att2(enc2, dec2)), dim=1)
        dec2 = self.spatial_attention2(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self.att1(enc1, dec1)), dim=1)
        dec1 = self.spatial_attention1(dec1)
        dec1 = self.decoder1(dec1)

        # Classifier
        output = self.cls(dec1)

        return output
