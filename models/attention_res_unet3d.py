"""
Attention Residual 3D UNet for CT segmentation
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution block with optional residual connection"""

    def __init__(self, in_channels, out_channels, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            out += identity

        out = self.relu(out)
        return out


class AttentionGate(nn.Module):
    """Attention gate for skip connections"""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Conv3d(gate_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv3d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate, skip):
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return skip * psi


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and attention"""

    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, out_channels // 2) if use_attention else None
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)

        if self.attention is not None:
            skip = self.attention(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionResUNet3D(nn.Module):
    """
    Attention Residual 3D UNet
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        feature_depths=[32, 64, 128, 256],
        use_attention=True,
        use_residual=True,
    ):
        super().__init__()

        self.encoder1 = ConvBlock(in_channels, feature_depths[0], use_residual=use_residual)
        self.encoder2 = ConvBlock(feature_depths[0], feature_depths[1], use_residual=use_residual)
        self.encoder3 = ConvBlock(feature_depths[1], feature_depths[2], use_residual=use_residual)
        self.encoder4 = ConvBlock(feature_depths[2], feature_depths[3], use_residual=use_residual)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(feature_depths[3], feature_depths[3] * 2, use_residual=use_residual)

        self.decoder4 = DecoderBlock(
            feature_depths[3] * 2, feature_depths[3], feature_depths[3], use_attention
        )
        self.decoder3 = DecoderBlock(
            feature_depths[3], feature_depths[2], feature_depths[2], use_attention
        )
        self.decoder2 = DecoderBlock(
            feature_depths[2], feature_depths[1], feature_depths[1], use_attention
        )
        self.decoder1 = DecoderBlock(
            feature_depths[1], feature_depths[0], feature_depths[0], use_attention
        )

        self.final_conv = nn.Conv3d(feature_depths[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        out = self.final_conv(d1)
        return out  # Return logits, BCEWithLogitsLoss will apply sigmoid internally


if __name__ == "__main__":
    model = AttentionResUNet3D(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
