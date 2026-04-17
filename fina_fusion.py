# =========================================================
# IMPORTS
# =========================================================

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset

# =========================================================
# STEP 1 — DATASET
# =========================================================

class MSRSDataset(Dataset):

    def __init__(self, root_dir):

        self.ir_dir = os.path.join(root_dir,"ir")
        self.vi_dir = os.path.join(root_dir,"vi")
        self.seg_dir = os.path.join(root_dir,"Segmentation_labels")

        self.files = sorted(os.listdir(self.ir_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        ir = cv2.imread(os.path.join(self.ir_dir,fname),0)
        vi = cv2.imread(os.path.join(self.vi_dir,fname))
        seg = cv2.imread(os.path.join(self.seg_dir,fname),0)

        # ======================
        # Preprocessing
        # ======================

        ir = cv2.resize(ir,(256,256))/255.0
        vi = cv2.resize(vi,(256,256))/255.0
        seg = cv2.resize(seg,(256,256))/255.0

        ir = torch.tensor(ir).unsqueeze(0).float()
        vi = torch.tensor(vi).permute(2,0,1).float()
        seg = torch.tensor(seg).unsqueeze(0).float()

        return ir,vi,seg


# =========================================================
# STEP 4 — Multi-scale CNN Feature Extractor
# =========================================================

class MultiScaleCNN(nn.Module):

    def __init__(self,in_channels):
        super().__init__()

        self.conv3 = nn.Conv2d(in_channels,32,3,padding=1)
        self.conv5 = nn.Conv2d(in_channels,32,5,padding=2)
        self.conv7 = nn.Conv2d(in_channels,32,7,padding=3)

    def forward(self,x):

        f1 = self.conv3(x)
        f2 = self.conv5(x)
        f3 = self.conv7(x)

        return torch.cat([f1,f2,f3],dim=1)


# =========================================================
# STEP 5 — Generator (Dual Branch)
# =========================================================

class Generator(nn.Module):

    def __init__(self,in_channels):
        super().__init__()

        self.feature = MultiScaleCNN(in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(96,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):

        f = self.feature(x)
        return self.conv(f)


# =========================================================
# STEP 6 — Attention Fusion
# =========================================================

class AttentionFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.att = nn.Sequential(
            nn.Conv2d(64,64,1),
            nn.Sigmoid()
        )

    def forward(self,f_ir,f_vi,seg):

        fused = torch.cat([f_ir,f_vi],dim=1)

        att = self.att(fused)

        # expand segmentation mask to all channels
        seg = seg.repeat(1,fused.shape[1],1,1)

        att = att * seg

        return fused * att


# =========================================================
# STEP 7 — Decoder (Reconstruction)
# =========================================================

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,1,3,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.conv(x)


# =========================================================
# LOAD DATA + MODELS
# =========================================================

root_dir = "MSRS"

dataset = MSRSDataset(root_dir)

gen_ir = Generator(1)
gen_vi = Generator(3)
attention = AttentionFusion()
decoder = Decoder()

# =========================================================
# RUN PIPELINE
# =========================================================

for i in range(2):

    ir,vi,seg = dataset[i]

    ir = ir.unsqueeze(0)
    vi = vi.unsqueeze(0)
    seg = seg.unsqueeze(0)

    # Generator features
    f_ir = gen_ir(ir)
    f_vi = gen_vi(vi)

    # Attention fusion
    fused_feature = attention(f_ir,f_vi,seg)

    # Decoder reconstruction
    fused_image = decoder(fused_feature)

    # =====================================================
    # PRINT OUTPUT SHAPES (COMMAND WINDOW)
    # =====================================================

    print("\nSample:",i)
    print("IR feature shape:",f_ir.shape)
    print("VI feature shape:",f_vi.shape)
    print("Fused feature shape:",fused_feature.shape)
    print("Final fused image shape:",fused_image.shape)

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    ir_feat = f_ir[0,0].detach().numpy()
    vi_feat = f_vi[0,0].detach().numpy()
    fused_feat = fused_feature[0,0].detach().numpy()
    fused_img = fused_image[0,0].detach().numpy()

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.imshow(ir_feat,cmap='gray')
    plt.title("IR Generator Feature")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(vi_feat,cmap='gray')
    plt.title("VI Generator Feature")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(fused_feat,cmap='gray')
    plt.title("Attention Fused Feature")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(fused_img,cmap='gray')
    plt.title("Final Fused Image (Decoder)")
    plt.axis("off")

    plt.show()
