import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import glob
import torch

class BasicDataset(Dataset):

    def __init__(self,patch_h,patch_w,datasetName,netType,train_mode = False):

        self.patch_h = patch_h
        self.patch_w = patch_w

        if netType == 'unet' or netType == 'deeplabv3plus':
            self.imgTrans = False
        else: 
            self.imgTrans = True

        self.transform = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
        ])    

        self.dataset = datasetName

        if datasetName == 'seam':
            self.n1 = 1006
            self.n2 = 782
            self.train_data_dir = '../data/classification/seamaiForTrain/input'
            self.train_label_dir = '../data/classification/seamaiForTrain/target'
            self.valid_data_dir = '../data/classification/seamaiForVal/input'
            self.valid_label_dir = '../data/classification/seamaiForVal/target'
        elif datasetName == 'salt':
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = '../data/detection/train/input'
            self.train_label_dir = '../data/detection/train/target'
            self.valid_data_dir = '../data/detection/valid/input'
            self.valid_label_dir = '../data/detection/valid/target'
        elif datasetName == 'fault_896':
            self.n1 = 896
            self.n2 = 896
            self.train_data_dir = '/home/zxguo/data/faultdataset/dataset_896/train/image'
            self.train_label_dir = '/home/zxguo/data/faultdataset/dataset_896/train/label'
            self.valid_data_dir = '/home/zxguo/data/faultdataset/dataset_896/valid/image'
            self.valid_label_dir = '/home/zxguo/data/faultdataset/dataset_896/valid/label'
        elif datasetName == 'crater':
            self.n1 = 1022
            self.n2 = 1022
            self.train_data_dir = '/home/zxguo/data/crater/crater/train/image'
            self.train_label_dir = '/home/zxguo/data/crater/crater/train/label'
            self.valid_data_dir = '/home/zxguo/data/crater/crater/valid/image'
            self.valid_label_dir = '/home/zxguo/data/crater/crater/valid/label'
        elif datasetName == 'das':
            self.n1 = 512
            self.n2 = 512
            self.train_data_dir = '/home/zxguo/data/das/dataset/train/image'
            self.train_label_dir = '/home/zxguo/data/das/dataset/train/label'
            self.valid_data_dir = '/home/zxguo/data/das/dataset/valid/image'
            self.valid_label_dir = '/home/zxguo/data/das/dataset/valid/label'
        else:
            print("Dataset error!!")
        print('netType:' + netType)
        print('dataset:' + datasetName)
        print('patch_h:' + str(patch_h))
        print('patch_w:' + str(patch_w))

        if train_mode:
            self.data_dir = self.train_data_dir
            self.label_dir = self.train_label_dir
        else:
            self.data_dir = self.valid_data_dir
            self.label_dir = self.valid_label_dir

        self.ids = len(os.listdir(self.data_dir))
    def __len__(self):
        return self.ids

    def __getitem__(self,index):
        
        dPath = self.data_dir+'/'+str(index)+'.dat'
        tPath = self.label_dir+'/'+str(index)+'.dat'
        data = np.fromfile(dPath,np.float32).reshape(self.n1,self.n2)
        label = np.fromfile(tPath,np.int8).reshape(self.n1,self.n2)

        data = np.reshape(data,(1,1,self.n1,self.n2))
        data = np.concatenate([data,self.data_aug(data)],axis=0)
        label = np.reshape(label,(1,1,self.n1,self.n2))
        label = np.concatenate([label,self.data_aug(label)],axis=0)

        if self.imgTrans:
            img_tensor = np.zeros([2,1,self.patch_h*14,self.patch_w*14],np.float32)
            for i in range(data.shape[0]):
                img = Image.fromarray(np.uint8(data[i,0]))
                img_tensor[i,0] = self.transform(img)
            data = img_tensor
            data = data.repeat(3,axis=1)
        elif not self.imgTrans:
            data = data/255

        return data,label

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.fliplr(np.squeeze(data))
        return data_fliplr.reshape(b,c,h,w)

class TestDataset(BasicDataset):

    def __init__(self,patch_h,patch_w,datasetName,netType,dino_pretrain = False):
        super().__init__(patch_h,patch_w,datasetName,netType,train_mode=False,dino_pretrain=dino_pretrain)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.dino_pretrain = dino_pretrain

        if netType == 'unet' or netType == 'deeplabv3plus':
            self.imgTrans = False
        else: 
            self.imgTrans = True

        self.transform = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
        ])    

        self.dataset = datasetName

        if datasetName == 'fault_512':
            self.n1 = 512
            self.n2 = 512
            self.data_dir = '/home/zxguo/data/faultdataset/test_512/'
        else:
            print("Dataset error!!")
        print('netType:' + netType)
        print('dataset:' + datasetName)
        print('patch_h:' + str(patch_h))
        print('patch_w:' + str(patch_w))

        self.ids = len(os.listdir(self.data_dir))
    def __len__(self):
        return self.ids

    def __getitem__(self,index):
        dPath = self.data_dir+'/'+str(index)+'.dat'
        data = np.fromfile(dPath,np.float32).reshape(self.n1,self.n2)
        data = np.reshape(data,(1,1,self.n1,self.n2))
        data = np.concatenate([data,self.data_aug(data)],axis=0)
        if self.imgTrans:
            img_tensor = np.zeros([2,1,self.patch_h*14,self.patch_w*14],np.float32)
            for i in range(data.shape[0]):
                img = Image.fromarray(np.uint8(data[i,0]))
                img_tensor[i,0] = self.transform(img)
            data = img_tensor
            if self.dino_pretrain:
                data = data.repeat(3,axis=1)
        elif not self.imgTrans:
            data = data/255

        return data

class FusionDataset(Dataset):
    def __init__(self,datasetName='fusion'):
        self.dataset = datasetName
        if datasetName == 'fusion':
            folders = ['../data/fusion/cgg','../data/fusion/slb']
            file_type = '*.dat'
            file_paths = []
            for folder in folders:
                file_paths.extend(glob.glob(f"{folder}/{file_type}"))
        self.ids = len(file_paths)
        self.dataPath = file_paths

    def __len__(self):
        return self.ids

    def __getitem__(self,index):
        dPath = self.dataPath[index]
        n1,n2 = int(dPath.split('.')[-2].split('_')[-2]),int(dPath.split('.')[-2].split('_')[-1])
        data = np.fromfile(dPath,np.float32).reshape(n1,n2)
        data = torch.from_numpy(np.reshape(data,(1,1,n1,n2)))
        data = self.interpolate(data,size=(data.shape[2]//14*14,data.shape[3]//14*14))
        data = self.norm(self.stdNorm(data))
        data = torch.squeeze(data.repeat(1,3,1,1),dim=0)
        return data

    def stdNorm(self,img):
        return (img - torch.mean(img))/torch.std(img)

    def norm(self,img):
        return (img - torch.min(img))/(torch.max(img) - torch.min(img))

    def interpolate(self,img,size):
        return  F.interpolate(img,size=size,mode='bilinear')
    
class RegressionDataset(Dataset):
    def __init__(self,datasetName='denoise',netType='setr1',train_mode = False):
        self.dataset = datasetName
        self.netType = netType
        if datasetName == "denoise":
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = '../data/denoise/train/input'
            self.train_label_dir = '../data/denoise/train/label'
            self.valid_data_dir = '../data/denoise/valid/input'
            self.valid_label_dir = '../data/denoise/valid/label'
        elif datasetName == "reflectivity":
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = '../data/reflectivity/train/input'
            self.train_label_dir = '../data/reflectivity/train/label'
            # self.valid_data_dir = '../data/reflectivity/valid/input'
            # self.valid_label_dir = '../data/reflectivity/valid/label'
            self.valid_data_dir = '../data/Inversion/SEAMseismic'
            self.valid_label_dir = '../data/Inversion/SEAMreflect'
        elif datasetName == "rgt":
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = '../data/rgt/train/input'
            self.train_label_dir = '../data/rgt/train/label'
            self.valid_data_dir = '../data/rgt/valid/input'
            self.valid_label_dir = '../data/rgt/valid/label'  
        elif datasetName == "rgt_ucf":
            self.n1 = 434
            self.n2 = 994
            self.train_data_dir = '/home/zxguo/data/rgt_ucf_434_994/train/input'
            self.train_label_dir = '/home/zxguo/data/rgt_ucf_434_994/train/label'
            self.valid_data_dir = '/home/zxguo/data/rgt_ucf_434_994/valid/input'
            self.valid_label_dir = '/home/zxguo/data/rgt_ucf_434_994/valid/label'                 
        if train_mode:
            self.data_dir = self.train_data_dir
            self.label_dir = self.train_label_dir
        else:
            self.data_dir = self.valid_data_dir
            self.label_dir = self.valid_label_dir
        self.ids = len(os.listdir(self.data_dir))

    def __len__(self):
        return self.ids

    def __getitem__(self,index):
        dPath = self.data_dir+'/'+str(index)+'.dat'
        tPath = self.label_dir+'/'+str(index)+'.dat'
        data = np.fromfile(dPath,np.float32).reshape(self.n1,self.n2)
        label = np.fromfile(tPath,np.float32).reshape(self.n1,self.n2)
        data = np.reshape(data,(1,1,self.n1,self.n2))
        label = np.reshape(label,(1,1,self.n1,self.n2))

        if self.netType == "deeplabv3plus" and self.dataset=="rgt_ucf":
            data = self.interpolate(data,(512,1024))
            label = self.interpolate(label,(512,1024))

        data = self.stdNorm(data)
        label = self.norm(label)
        data = np.concatenate([data,self.data_aug(data)],axis=0)
        label = np.concatenate([label,self.data_aug(label)],axis=0)

        if self.netType != "unet" and self.netType != "deeplabv3plus":
            data = data.repeat(3,axis=1)
        
        return data,label

    def stdNorm(self,img):
        return (img - np.mean(img))/(np.std(img)+1e-6)

    def norm(self,img):
        return (img - np.min(img))/(np.max(img) - np.min(img)+1e-6)

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.fliplr(np.squeeze(data))
        return data_fliplr.reshape(b,c,h,w)
    
    def interpolate(self,data,size):
        data_pt = torch.from_numpy(data)
        data_pt = F.interpolate(data_pt,size=size,mode="bilinear")
        return data_pt.numpy()

def loadMarmousiData():
    n1,n2 = 751,2301
    dataPath = "../data/inversion/marmousi/"
    sx = np.fromfile(dataPath+"sxr.dat",np.float32).reshape(n2,n1).T
    wx = np.fromfile(dataPath+"wx.dat",np.float32)# 101 samplings---ricker
    mask = np.fromfile(dataPath+"mask.dat",np.float32).reshape(n2,n1).T
    px = np.fromfile(dataPath+"px.dat",np.float32).reshape(n2,n1).T
    return sx,px,mask,wx    

if __name__ == '__main__':

    train_set = DenoiseDataset(datasetName='reflectivity')
    print(train_set.__getitem__(0)[1].shape)
    print(len(train_set))
    # print(train_set.__getitem__(0)[1].shape)
    plt.imshow(train_set.__getitem__(4999)[0][0,0,:,:],cmap='gray')
    plt.colorbar()
    plt.savefig("data.png")
    plt.figure()
    plt.imshow(train_set.__getitem__(4999)[1][0,0,:,:],cmap='gray')
    plt.colorbar()
    plt.savefig("label.png")

    # train_set = BasicDataset(72,56,'seam','setr1',True,True)
    # print(train_set.__getitem__(0)[1].shape)
    # print(len(train_set))
    # # print(train_set.__getitem__(0)[1].shape)
    # plt.imshow(train_set.__getitem__(20)[0][0,0,:,:],cmap='gray')
    # plt.colorbar()
    # plt.savefig("data.png")
    # plt.imshow(train_set.__getitem__(20)[1][0,0,:,:],cmap='jet')
    # plt.colorbar()
    # plt.savefig("label.png")

    mask = np.fromfile("./data/inversion/marmousi/mask.dat",np.float32).reshape(2301,751).T
    plt.imshow(mask)
    plt.savefig("mask.jpg")
