class DAAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAAM, self).__init__()
        self.conv = DWConv(in_channels, out_channels, k=3, s=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv(x)
        attn = self.attention(conv_out)
        return conv_out * attn

# إضافة DAAM لـ globals
globals()['DAAM'] = DAAM

from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

model.train(data="lastYaml.yaml", epochs=50, batch=16, imgsz=640)

