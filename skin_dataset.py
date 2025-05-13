from torch.utils.data import Dataset,DataLoader
import glob
import os
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# 图像变换
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(30),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 掩码变换（不需要归一化）
mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class SkinDataset(Dataset):
    def __init__(self, data_root, subset='train', transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        im_root = ""
        mask_root = ""
        if subset == 'train':
            im_root = "../ISIC_Challenge_2016/Part1/ISBI2016_ISIC_Part1_Training_Data"
            mask_root = "../ISIC_Challenge_2016/Part1/ISBI2016_ISIC_Part1_Training_GroundTruth"
        else:
            im_root = "../ISIC_Challenge_2016/Part1/ISBI2016_ISIC_Part1_Test_Data"
            mask_root = "../ISIC_Challenge_2016/Part1/ISBI2016_ISIC_Part1_Test_GroundTruth"

        # 检索所有图片文件名
        self.all__samples = []

        all_imfiles = glob.glob(os.path.join(im_root,"*.jpg"))
        for imf in all_imfiles:
            maskf = os.path.join(mask_root,os.path.basename(imf).replace(".jpg","_Segmentation.png"))
            if os.path.exists(maskf):
                self.all__samples.append([imf,maskf])
            
        print("total samples:",len(self.all__samples))

    def __len__(self):
        return len(self.all__samples)
    
    def __getitem__(self,idx):
        imf,maskf = self.all__samples[idx]
        image = cv2.imread(imf)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(maskf, cv2.IMREAD_GRAYSCALE)
        
        # 分别对图像和掩码进行变换
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        mask[mask>0] = 1
        return image, mask

if __name__ == '__main__':
    dataset = SkinDataset("../ISIC_Challenge_2016/Part1/ISBI2016_ISIC_Part1_Training_Data")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    for img, msk in dataloader:
        print(img.shape, msk.shape)
        img = img[0,...].numpy().transpose((1,2,0))
        msk = msk[0,0,...].numpy()  # 取第一个通道，去掉batch维度
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(msk, cmap="gray")
        plt.savefig("skin_lesion_segmentation.png")
        break
            
            
            