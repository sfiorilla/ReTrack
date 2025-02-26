import os
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Utils
def print_images( img, mask ):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title("Image to Save")
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Mask to save")
    ax[1].axis('off')
    plt.show()
    
    
# select the device for computation
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
############

#initialize a segmentation model
maskNet_transform =DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms().to('cuda')

################################################################
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, image_dir,preprocess=maskNet_transform ):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.preprocess = preprocess
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        img=resize_function(img,(128,64))
        y = transforms.ToTensor()(img)# normal image 
        x = self.preprocess(img)
        name = os.path.basename(image_path) 
        return x,name,y
    
    
# Imposta il DataLoader

dataset_folder = "./MOT17_ReID/bounding_box_train"
output_folder = "./MOT17_ReID/bounding_box_train_dlv3"
  
shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)

batch_size = 16
dataset = ImageDataset(dataset_folder)

def my_fun(batch):
    print(f"batch is a {type(batch)} of size{len(batch)} and each element is a {type(batch[0])}")
    for i in range(len(batch)):
        a,b,c = batch[i]
        print(a.shape)
        print(b)
        print(c.shape)
        print("-------------------------")


    assert False, "stop"
    return batch

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False  )

print("Inititialized dataloader..")

from torchvision.transforms.functional import to_pil_image
import traceback

maskNet = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
maskNet.eval()

maskNet = torch.nn.DataParallel(maskNet)
maskNet = maskNet.to(device)


weights = DeepLabV3_ResNet101_Weights.DEFAULT
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
PERSON = class_to_idx["person"] #is number 15

##################################
#Utils
resize_function = lambda x, size: transforms.Resize(size)(x)
print_tensor = lambda txt,x: print(f"{txt} shape: {x.shape}, max: {x.max().item()}, min: {x.min().item()}")

print("model device in use is: ",next(maskNet.parameters()).device.type)


def save_files(output_folder, masks, imgs, fnames):
    
    for i, f in enumerate(fnames):
        try:
            mm = Image.fromarray( masks[i].squeeze() ).convert("L") # from CHW to HW, L is "greyscale"
            ii = Image.fromarray( imgs[i].transpose(1, 2, 0)).convert("RGB") # from CHW to HWC
            
            mask_path = os.path.join(output_folder, f.replace(".jpg","_mask.png"))
            mm.save(mask_path, format="PNG" )
            img_path = os.path.join(output_folder, f.replace(".jpg",".png"))
            ii.save(img_path, format="PNG" )
            #print_images(ii,mm)
            
        except Exception as e:
                print(f"Error for line {i} and fname is {f}")
                print(f"An error occurred: {e}")
                traceback.print_exc()
                assert False, "stop for Exception"

                

counter_continue=0
for idx, (images,fnames, original_imgs) in enumerate(dataloader):
    
    print( f"[{idx}/{len(dataloader)}]")
    inputs = images.to(device)
    with torch.no_grad():
        logits = maskNet(inputs)["out"]
        preds = logits.softmax(dim=1).argmax(dim=1, keepdim=True)
        person_mask = (preds==PERSON).float()
        mask_resized = F.interpolate(person_mask, size=(128,64), mode='bilinear', align_corners=True)
        
    np_masks = mask_resized.cpu().numpy()
    ims_no_bg = original_imgs.cpu().numpy() * np_masks
    
    save_files( output_folder, (255*np_masks).astype(np.uint8), (255*ims_no_bg).astype(np.uint8), fnames)
    

print("scanning is complete")
