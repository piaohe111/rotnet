import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
import os
from PIL import Image
class rotdata(dataset):
    def __init__(self,imagespath,train_ratio):
        super(rotdata,self).__init__()
        self.data_preprose=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ])
        imagenames=os.listdir(imagespath)
        train_imagenums=int(len(imagenames)*train_ratio)
        self.train_imgs_labels=[]
        for imagename in imagenames[:train_imagenums]:
            imgs=[]
            img=Image.open(os.path.join(imagespath,imagename))
            imgs+=[img]
            for angle in [90,180,270]:
                img_rotate=img.rotate(angle)
                imgs+=[img_rotate]
            labels=[0,1,2,3]
            self.train_imgs_labels+=list(zip(imgs,labels))
        self.vali_imgs_labels = []
        for imagename in imagenames[train_imagenums:]:
            imgs=[]
            img=Image.open(os.path.join(imagespath,imagename))
            imgs+=[img]
            for angle in [90,180,270]:
                img_rotate=img.rotate(angle)
                imgs+=[img_rotate]
            labels=[0,1,2,3]
            self.vali_imgs_labels+=list(zip(imgs,labels))
    def __getitem__(self, index):
        img,label=self.train_imgs_labels[index]
        return self.data_preprose(img),label
    def __len__(self):
        return len(self.imgs_labels)
