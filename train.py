from data import rotdata
import argparse
import torch
import torch.utils.data.dataloader as dataloader
from torchvision.models import resnet50,shufflenet_v2_x2_0,mobilenet_v2
import torch.nn as nn
from
def parse_args():
    parse=argparse.ArgumentParser(description="get image rotate angle")
    parse.add_argument("--imagepath",type=str)
    parse.add_argument("--batch_size",default=1024,type=int)
    parse.add_argument("--backbone",type=str)
    parse.add_argument("--lr", default=1e-3)
    return parse.parse_args()

def getBackbone(backbone,backbones):
    assert backbone in backbones
    if(backbone=="resnet50"):
        model=resnet50(pretrained=True)
        model.fc=torch.nn.Linear(in_features=...,out_features=4)
    if(backbone=="shufflenet_v2_x2_0"):
        model=shufflenet_v2_x2_0(pretrained=True)
        model.fc = torch.nn.Linear(in_features=..., out_features=4)
    if(backbone=="mobilenetv2"):
        model=mobilenet_v2(pretrained=True)
        model.classifier = torch.nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=...,out_features=4),
        )
    #if(backbone=="mobilenetv3"):

    return model
backbones=["resnet50","shufflenet_v2_x2_0","mobilenetv2","mobilenetv3","ghostnet","efficientnet"]
args=parse_args()
imagepath=args.imagepath
batch_size=args.batch_size
backbone=args.backbone
lr=args.lr
imgdata=rotdata(args.imagepath,0.8)
vali_data=imgdata.vali_imgs_labels
train_data_iter=dataloader(imgdata,batch_size=batch_size,shuffle=True,num_workers=1)
train_model=getBackbone(backbone,backbones)
optimizer=torch.optim.SGD(train_model.parameters(),lr=lr,weight_decay=5e-4,momentum=0.9)
loss=torch.nn.CrossEntropyLoss()

for index,img_label in enumerate
