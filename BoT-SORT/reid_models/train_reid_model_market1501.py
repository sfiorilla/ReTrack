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

from pifu.lib.options import BaseOptions
from pifu.lib.model import HGPIFuNet

import argparse

parser = argparse.ArgumentParser(description="need losses weights to execute")

parser.add_argument('--wt', type=float, required=True, help='weight of triplet loss')
parser.add_argument('--wx', type=float, required=True, help='weight of softmax loss')
parser.add_argument('--wc', type=float, required=True, help='weight of center loss')
parser.add_argument('--wr', type=float, required=True, help='weight of reconstruction loss')

parser.add_argument('--bs', type=int, required=True, help='batch size')
parser.add_argument('--k',  type=int, required=True, help='K for sampling people')
parser.add_argument('--ep', type=int, required=True, help='epocs')
parser.add_argument('--lr', type=float, required=True, help='Learning Rate')
parser.add_argument("--scheduler", type=str,required=True, help="Scheduler steps" )

parser.add_argument('--restore',   type=str, required=True, help='if restore the model')
parser.add_argument('--evalfreq',  type=int, required=True, help='Evaluation frequency')
parser.add_argument('--printfreq', type=int, required=True, help='Print loss frequency')
parser.add_argument('--onlytest',  type=str, required=True, help='Apply Only Test')
parser.add_argument('--model',  type=str, required=False, help='Model name')
parser.add_argument('--dist',  type=str, required=False, help='options: cosine or euclidean')
parser.add_argument('--neck',  type=str, required=False, help='options: before, after, concat, avg, sum')

args = parser.parse_args()

wt = args.wt
wx = args.wx
wc = args.wc
wr = args.wr
BS =args.bs
K =args.k
EPOCHS=args.ep
LR=args.lr


steps = args.scheduler.split(" ")
SCHEDULER = int(steps[0]) if len(steps)>1 else [int(x) for x in steps]
MODELNAME=args.model if args.model else "full_model.pth.tar"
RESUME=args.restore
EVALFREQ=args.evalfreq
PRINTFREQ=args.printfreq
ONLYTEST = args.onlytest.strip().lower() == 'true'
NECK= args.neck.strip().lower() if args.neck else 'after'
DIST=args.dist.strip().lower() if args.dist else 'cosine'


print(f"ARGS: {args}")

########## CREATE THE MODEL #######


import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet_IBN', 'resnet50_ibn_a', 'resnet101_ibn_a',
           'resnet152_ibn_a']


model_urls = {
    'r50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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

    def __init__(self, num_classes, last_stride=1, pretrained=not RESUME, neck='bnneck', neck_feat=NECK, model_name='resnet50_ibn_a', pretrain_choice='imagenet'):
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
            
            if self.neck=='bnneck':
                if self.neck_feat=='concat':
                    return torch.cat((global_feat, feat), dim=1)  # Result is (bs, 4096)

                if self.neck_feat=='reverse_concat':
                    return torch.cat((global_feat, feat), dim=1)
                if self.neck_feat=='sum':
                    return global_feat+feat
                if self.neck_feat=='mean':
                    return (global_feat+feat)/2

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


###### data augmentation pipeline ######## 

import math
import random


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


input_size=(128,256) #width,height
mu = (0.48145466, 0.4578275, 0.40821073)
sigma = (0.26862954, 0.26130258, 0.27577711)
normalize_transform = transforms.Normalize(mean=mu, std=sigma)

train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(padding=10),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        normalize_transform,
        RandomErasing(probability=0.5, mean=mu)
    ])

valid_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize_transform
        # transforms.Lambda(lambda x:x/torch.max(x))
    ])

datamanager = torchreid.data.ImageDataManager(
    root=".",
    sources="market1501",
    targets="market1501",
    height=256,
    width=128,
    batch_size_train=BS,
    batch_size_test=100,
    #transforms=train_transform,
    num_instances=K,
    train_sampler='RandomIdentitySampler',
)

datamanager.transform_tr=train_transform
datamanager.transform_te=valid_transform




######## Losses #######

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt() #for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss= self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum()/batch_size

        #dist = []
        #for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        #dist = torch.cat(dist)
        #loss = dist.mean()

        return loss


class buildPifuExtractor:
    
    def __init__(self, opt, projection_mode='orthogonal', stopMaskNet=True):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize((self.opt.resolution ,self.opt.resolution//2)),#(h256,w128)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        
        # create net
        netG = HGPIFuNet(opt, projection_mode)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))
            print('Loaded checkpoints for', netG.name)

        self.cuda = cuda
        self.netG = netG #if device_count()<2 else nn.DataParallel(netG)
        self.netG = self.netG.to(cuda)
        self.netG.eval()
        
        if not stopMaskNet:
            print('Initializing pretrained DeepLabV3 model')
            self.maskNet = deeplabv3_resnet101(pretrained=True)
            self.maskNet = self.maskNet if device_count() <2 else nn.DataParallel(self.maskNet)
            self.maskNet = self.maskNet.to(device=cuda)
            self.maskNet.eval()
            print('DeepLabV3 is ready')
        else:
            self.maskNet=None
            print('No mask net was loaded')        
    
    def process_image( self, image, mask_path=None):
        image = transforms.Resize(self.load_size)(image)
        image = image.to(device=cuda)

        # Mask
        if mask_path is None:
            mask = self.maskNet(image)['out'].argmax(1).float().to(device=cuda)                                                                       
        else:                                                                 
            mask = Image.open(mask_path).convert('L')
            mask = transforms.ToTensor()(mask).float()   
            
        #Apply the RESIZED mask to the image 
        mask = transforms.Resize(self.load_size)(mask) 
        image = mask.unsqueeze(1) * image
        return image
    
    def load_image(self, image_path, mask_path=None):
        # Read image and name from path
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert('RGB')
        image = self.to_tensor(image).to(device=cuda).unsqueeze(0)
        
        return self.process_image(image,mask_path)
        
        
    def evaluation(self, images, use_octree=False):
        with torch.no_grad():
            image_tensor = images.to(device=cuda)
            # image_tensor = data['img'].to(device=cuda)
            # calib_tensor = data['calib'].to(device=cuda)
            self.netG.filter(image_tensor)
            batch_tensor = torch.stack(self.netG.im_feat_list, dim=0)
            return batch_tensor.squeeze()  
    
            

sys.argv = [sys.argv[0]] 
opt = BaseOptions().parse()
opt.load_netG_checkpoint_path = './pifu/checkpoints/net_G'
opt.resolution = 256 
cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu') 

        
import time
from functools import wraps

def timing_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"'{func.__name__}'executed in: {elapsed_time:.2f} secs.")
        return result
    return wrapper


class buildPifuMasks:
    class SimpleDataset(Dataset):
        def __init__(self, transform, root_dir='./market1501/Market-1501-v15.09.15/bounding_box_train_nobg/'):
            self.root_dir = root_dir
            self.transform = transform
            self.flist = os.listdir(root_dir)

        def __len__(self):
            return len(self.flist)

        def __getitem__(self, idx):
            img_name = self.flist[idx]
            img_path = os.path.join(self.root_dir, img_name)
            pil_image = Image.open(img_path).convert('RGB')
            img = self.transform(pil_image)
            return img, img_name[:-4] 

    @timing_wrapper
    def populate_dictionary(self,bs=8):
        dataset = self.SimpleDataset(transform=self.pifuEncoder.to_tensor)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
        for idx,(batch_images, batch_names) in enumerate(dataloader):
            outputs = self.pifuEncoder.evaluation(batch_images)
            #outputs shape is [bs,c=256,h=64,w=32]
            for name, output in zip(batch_names, outputs):
                self.my_dict[name] = output.mean(dim=0, keepdim=True)
        
    def __init__(self, pifu_encoder):
        self.pifuEncoder = pifu_encoder
        self.my_dict = {}
        print('Precomputing Pifu features..')
        self.populate_dictionary()



class ReconstructionLoss(nn.Module):
    """
        Reconstruction loss.
    """
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(ReconstructionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.pifuEncoder= buildPifuExtractor(opt) #model is send to cuda inside the builder constructor
        #self.nobgs = self.__build_preprocessed_data()
        # Freeze pifuEncoder parameters
        for param in self.pifuEncoder.netG.parameters():
            param.requires_grad = False 
        
        self.gt_masks_dict={}
        if not ONLYTEST:
            myObj = buildPifuMasks(self.pifuEncoder)
            self.gt_masks_dict = myObj.my_dict
        
        
    def forward(self, m_preds, paths):

        """
        Args:
            m_preds: Predicted masks from features.
            paths: List of images file paths from which IDs will be extracted.
               (batch_size, C=256, W=256, H=128) 
        """

        extract_id = lambda x : x.split('/')[-1][:-4]
        m_gt = torch.stack(
                [self.gt_masks_dict[ extract_id(f)] for f in paths ]
            ,dim=0)

        #images=self.pifuEncoder.process_image(images)
        #imsnobgs = [self.nobgs[extract_id(f)] for f in paths]
        #batch_imgs = torch.stack([self.pifuEncoder.to_tensor(img) for img in imsnobgs])
        #m_pifu = self.pifuEncoder.evaluation(batch_imgs)
        
        H,W= m_preds.shape[-2], m_gt.shape[-1]

        #reduce along channels to compute a weighted mask
        #m_gt = m_pifu.mean(dim=1, keepdim=True)
        m_preds = m_preds.mean(dim=1, keepdim=True)
        pixel_loss = F.smooth_l1_loss(m_preds, m_gt, reduction='none')

        #sum all the pixel and normalize by num of pixel (H*W)
        loss = pixel_loss.sum(dim=(2, 3)) / (H * W)

        return loss.sum()
       


class CostumEngine(engine.Engine):
    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        weight_c=1,
        weight_r=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        filepath=None,
        load_optimizer2=False
    ):
        super(CostumEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.filepath = filepath
       

        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0 and weight_c >= 0 and weight_r>=0
        assert weight_t + weight_x + weight_c +weight_r > 0

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_c = weight_c
        self.weight_r = weight_r

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = losses.CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

        self.criterion_c= CenterLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu
        )
        self.optimizer2= torch.optim.SGD(self.criterion_c.parameters(), lr=0.5)

        if load_optimizer2:
            assert self.filepath, "Load_model is True but filepath is Empty"
            checkpoints=torch.load(self.filepath)
            opt2_val=checkpoints.get("optimizer2_state_dict")
            if opt2_val:
                self.optimizer2.load_state_dict(opt2_val)
            else:
                print('Model loaded but no values for optimizer2 of center loss.')        

        self.criterion_r = ReconstructionLoss(self.use_gpu)

        self.save_model = self.build_save_model_fun()
        
    #overriding the function of save_model
    def build_save_model_fun(self):
        def new_fun(epoch, rank1=None, save_dir=None,is_best=False):
            #do not change the signature of the function
            dir_path = os.path.dirname(self.filepath)
            os.makedirs(dir_path, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer1_state_dict': self.optimizer.state_dict(),
                'optimizer2_state_dict': self.optimizer2.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'last_epoch': epoch+1
            }, self.filepath)
            print(f"Checkpoint saved to {self.filepath}...")
        return new_fun

    


    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
        
        if self.use_gpu:
            imgs = imgs.to(device=cuda)
            pids = pids.to(device=cuda)
        

        (outputs,features), shapefeats = self.model(imgs)
        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = (self.weight_t * loss_t.item() )

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = self.weight_x * loss_x.item()


        if self.weight_c >0:
            loss_c = self.compute_loss(self.criterion_c, features, pids)
            loss += self.weight_c * loss_c
            loss_summary['loss_c'] = self.weight_c*loss_c.item()

        if self.weight_r >0:
            loss_r = self.compute_loss(self.criterion_r, shapefeats, data["impath"] )
            loss += self.weight_r * loss_r
            loss_summary['loss_r'] = self.weight_r*loss_r.item()
            
        assert loss_summary
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        loss.backward()
        self.optimizer.step()
        
        for param in self.criterion_c.parameters():
            param.grad.data *= (1. / self.weight_c)

        self.optimizer2.step()

        return loss_summary
    
    def extract_features(self, input):
        ( feat_vecs, feat_masks ) = self.model(input)
        return feat_vecs
        



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
        #base_model = torchreid.models.build_model(name='resnet50_ibn_a', num_classes=datamanager.num_train_pids, loss="triplet")
        base_model=Baseline( num_classes=datamanager.num_train_pids )
        self.encoder = Encoder(base_model)
        self.decoder = Decoder(in_channels=2048, layers_number=2)
        self.decoder.apply(weights_init_kaiming)
            
    def forward(self, x):
        x, ext_feat = self.encoder(x)
        shape_feat = self.decoder(ext_feat)
        #x is a tuple of classification+features from the model output
        return x,shape_feat 
    
            
model = FullModel()

print(f"Currently running on {device_count()} gpus.")
model = model if device_count() <2 else nn.DataParallel(model)
model = model.cuda()

#optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=LR)

weight_decay = 0.0005
weight_decay_bias = 0.0005
biar_lr_factor = 1
params = []
for key, value in model.named_parameters():
    if not value.requires_grad:
        continue
    if "bias" in key:
        lr = LR * biar_lr_factor
        weight_decay = weight_decay_bias
    params += [{"params": [value], "lr": LR, "weight_decay": weight_decay}]

optimizer = getattr(torch.optim, "Adam")(params)
### created ADAM optizer


scheduler = torchreid.optim.build_lr_scheduler(optimizer,
    lr_scheduler='single_step' if isinstance(SCHEDULER, int) else 'multi_step',
    stepsize=SCHEDULER
)

#scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='multi_step', stepsize=[20,48,90] )

def save_model_state(model, opt1, opt2, scheduler, last_epoch, filepath):
    dir_path = os.path.dirname(filepath)
    os.makedirs(dir_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer1_state_dict': opt1.state_dict(),
        'optimizer2_state_dict': opt2.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'last_epoch': last_epoch
    }, filepath)

    print(f"Checkpoint saved to {filepath}")



def load_model_state(model, opt, scheduler, filepath):

    if not os.path.isfile(filepath):
        print(f"No checkpoint file found at: {filepath}")
        return

    checkpoints= torch.load(filepath)
    model.load_state_dict(checkpoints["model_state_dict"] )
    
    opt.load_state_dict(checkpoints["optimizer1_state_dict"] )
    #Note: optmizer2 (center loss) will be loaded inside the engine class
    scheduler.load_state_dict(checkpoints["scheduler_state_dict"] )
    last_epoch=checkpoints['last_epoch']

    print("Model state was successfully loaded")
    return last_epoch



SAVEDIR="./saved_weights/"
#SAVEDIR="./log/model/"

print(f"Resume {RESUME}, Eval Frequency {EVALFREQ}")

lepoch=0
if RESUME == 'TRUE':
    lepoch = load_model_state(model, optimizer,scheduler, SAVEDIR + MODELNAME)
    assert lepoch<EPOCHS or ONLYTEST, 'Last trained epoch is greater than EPOCHS value'


engine = CostumEngine(
    datamanager, model, optimizer,
    margin=0.3,
    weight_t=wt,
    weight_x=wx,
    weight_c=wc,
    weight_r=wr,
    scheduler=scheduler,
    filepath=SAVEDIR+MODELNAME,
    load_optimizer2=False #(RESUME=='TRUE')
)


print('\nValues inside () are AVG values')
engine.run(
    max_epoch=EPOCHS,
    save_dir=SAVEDIR,
    print_freq=PRINTFREQ,
    eval_freq=EVALFREQ,
    test_only=ONLYTEST,
    start_epoch=lepoch,
    dist_metric=DIST,
    visrank=ONLYTEST
)

#Check test with reranking
engine.run(
    test_only=True,
    rerank=True,
    dist_metric=DIST
)

#save_model_state(model, optimizer, engine.optimizer2, scheduler, EPOCHS, SAVEDIR+MODELNAME)

