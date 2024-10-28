import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // r, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // r, in_channels, kernel_size=1)
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1)  # Global Average Pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

# Res2Net Block
class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scaling=4):
        super(Res2NetBlock, self).__init__()
        self.scaling = scaling
        self.split_channels = out_channels // scaling
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels if i == 0 else self.split_channels, 
                                              self.split_channels, kernel_size=3, padding=1) 
                                    for i in range(scaling)])
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        splits = []
        for i in range(self.scaling):
            if i == 0:
                splits.append(F.relu(self.convs[i](x)))
            else:
                splits.append(F.relu(self.convs[i](splits[-1])))
        out = torch.cat(splits, dim=1)
        out = self.bn(self.conv_out(out))
        return out

# Double Conv Block with Res2Net and SE
class Res2NetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scaling=4, r=8):
        super(Res2NetSEBlock, self).__init__()
        self.res2net = Res2NetBlock(in_channels, out_channels, scaling)
        self.se = SEBlock(out_channels, r)
    
    def forward(self, x):
        out = self.res2net(x)
        out = self.se(out)
        return out

# UNet Model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
       # Encoder
        self.encoder1 = Res2NetSEBlock(in_channels, 32) # 64
        self.encoder2 = Res2NetSEBlock(32, 64)         #(64, 128)
        self.encoder3 = Res2NetSEBlock(64, 128)        #(128, 256)
        self.encoder4 = Res2NetSEBlock(128, 256)        #(256, 512)
        self.encoder5 = Res2NetSEBlock(256, 512)       #(512, 1024)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) #(1024, 512)
        self.decoder4 = Res2NetSEBlock(512, 256)                             #(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  #(512, 256
        self.decoder3 = Res2NetSEBlock(256, 128)                              #(512, 256
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  #(256, 128)
        self.decoder2 = Res2NetSEBlock(128, 64)                              #(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   #(128, 64)
        self.decoder1 = Res2NetSEBlock(64, 32)                               #(128, 64)
        
        # Final Output Layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)        # 64
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)  # 256x256 -> 256x256x64
        enc2 = self.encoder2(self.pool(enc1))  # 128x128x64 -> 128x128x128
        enc3 = self.encoder3(self.pool(enc2))  # 64x64x128 -> 64x64x256
        enc4 = self.encoder4(self.pool(enc3))  # 32x32x256 -> 32x32x512
        enc5 = self.encoder5(self.pool(enc4))  # 16x16x512 -> 16x16x1024

        # Decoder path
        dec4 = self.upconv4(enc5)  # 16x16x1024 -> 32x32x512
        dec4 = torch.cat((enc4, dec4), dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)  # 32x32x1024 -> 32x32x512
        
        dec3 = self.upconv3(dec4)  # 32x32x512 -> 64x64x256
        dec3 = torch.cat((enc3, dec3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)  # 64x64x512 -> 64x64x256
        
        dec2 = self.upconv2(dec3)  # 64x64x256 -> 128x128x128
        dec2 = torch.cat((enc2, dec2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)  # 128x128x256 -> 128x128x128
        
        dec1 = self.upconv1(dec2)  # 128x128x128 -> 256x256x64
        dec1 = torch.cat((enc1, dec1), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)  # 256x256x128 -> 256x256x64
        
        # Final output layer
        out = self.final_conv(dec1)  # 256x256x64 -> 256x256x1
        return torch.sigmoid(out)  # Sigmoid activation for binary segmentation

# Create model
#model = UNet(in_channels=1, out_channels=1)
#print(model)
