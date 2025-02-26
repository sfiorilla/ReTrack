import tempfile
import requests
import sys
import os

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import device_count
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101

import torchreid
from torchreid import metrics, losses, engine

#from reid_models.pifu.lib.options import BaseOptions
#from reid_models.pifu.lib.model import HGPIFuNet


##   GLOBAL VARIABLES
SAVEDIR="./reid_models/saved_weights/"
NECK='after'
DIST='cosine'

import math
import random

input_size=(128,256) #width,height
mu = (0.48145466, 0.4578275, 0.40821073)
sigma = (0.26862954, 0.26130258, 0.27577711)
normalize_transform = transforms.Normalize(mean=mu, std=sigma)

valid_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize_transform
        
    ])



########## STRUCTURE OF THE MODEL #######

model_urls = { 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth' }

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def load_weights_from_url(model, url):
    print('Downloading RN50 weights from ',url)
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    
    model.load_param(temp_file_path)
    print("Downloaded model.\n\n")




class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride=1, pretrained=False, neck='bnneck', neck_feat=NECK, model_name='resnet50_ibn_a', pretrain_choice='imagenet'):
        super(Baseline, self).__init__()
        
        self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])

        if pretrained:
            load_weights_from_url(self.base, model_urls["resnet50"] )
        
        #self.in_planes=2048
        self.feat_dims_l4 = self.base.layer4[-1].conv2.out_channels

        #if pretrain_choice == 'imagenet':
        #    self.base.load_param(model_path)
        #    print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo

        elif self.neck == 'bnneck':

            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)



    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        
        if not self.training:

            if self.neck_feat=='before':
                return global_feat

            if self.neck_feat=='after':
                return feat
            
            raise ValueError(f"Invalid values for self.neck_feat={self.neck_feat} or self.neck={self.neck}.")

        else:
            cls_score = self.classifier(feat)
            return cls_score, global_feat 

        # global feature for triplet loss
        #else:
        #if self.neck_feat == 'after':
        # print("Test with feature after BN")
        # return feat
        #else:
          # print("Test with feature before BN")
        # return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


class UpSamplingModule(nn.Module):
    def __init__(self, in_channels, is_last_layer=False):
        
        super(UpSamplingModule, self).__init__()
        layers = []

        # ConvTranspose2d: Double the dimension
        layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
        layers.append(nn.BatchNorm2d(in_channels // 2))
        layers.append(nn.ReLU())

        # Conv2d: keep dividing the channels for H and W dims
        layers.append(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1))
        if not is_last_layer:
            layers.append(nn.BatchNorm2d(in_channels // 2))
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):

    def __init__(self, in_channels, layers_number=2):
        super(Decoder, self).__init__()
        self.up_blocks = nn.ModuleList()

        for i in range(layers_number):
            self.up_blocks.append( UpSamplingModule(in_channels, is_last_layer=(i==layers_number-1) ))
            in_channels = in_channels // 2 #Divide the channels no. for the next block

    def forward(self, x):
        for up_block in self.up_blocks:
            x = up_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, base_model):
        super(Encoder, self).__init__()
        self.base_model = base_model
        #self.feat_dims = base_model.layer4[-1].conv2.out_channels
        self.feat_dims = base_model.feat_dims_l4

    def forward(self, x):
        f = self.base_model.base(x)
        x = self.base_model(x)
        return x, f


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        base_model=Baseline( num_classes=751 )
        self.encoder = Encoder(base_model)
        self.decoder = Decoder(in_channels=2048, layers_number=2)
        self.decoder.apply(weights_init_kaiming)
            
    def forward(self, x):
        x, ext_feat = self.encoder(x)
        shape_feat = self.decoder(ext_feat)
        #x is a tuple of classification+features from the model output
        return x,shape_feat 
    
from reid_models.model_interface import BaseInferenceModel

class M1(BaseInferenceModel):
    def __init__(self, con_file, wpath=os.getcwd()+"/reid_models/saved_weights/full_model.pth.tar",  dev='cuda', bs=8, *args, **kwargs ):
        self.use_gpu=True
        super().__init__(weights_path=wpath, config_file=con_file, device=dev, batch_size=bs, *args, **kwargs)
        

    def build_model(self, configs):
        return FullModel() 
        
    
    def load_reid_model(self, model, path ):
        
        if not os.path.isfile(path):
            print(f"No checkpoint file found at: {path}")
            return
        
        try:
            checkpoints= torch.load(path, map_location='cpu')
            state_dict=checkpoints["model_state_dict"]
            #remove module prefix from weights
            filtered_dict = {k: v for k, v in state_dict.items() if "module.encoder.base_model.classifier" not in k}
            new_state_dict = {k.replace("module.", ""): v for k, v in filtered_dict.items()}

            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print("Missing keys:", missing_keys)  
            print("Unexpected keys:", unexpected_keys)  
        
        except Exception as e:
            print(f"Error on loading checkpoints {e}")
            sys.exit(1)
            
        
    def extract_features(self, model, patches ):
        return model(patches)[0] #In test mode the model retursn a tuple (embs, reconstruction_embs), so embs at in pos 0
        


