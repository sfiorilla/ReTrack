from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch
from torch import nn
from torch.cuda import device_count

def postprocess(features):
    # Normalize feature to compute cosine distance
    #features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


from fast_reid.fastreid.config import get_cfg


#The Base class can be reused by every models to adapt to Bot-Sort Algoritm
class BaseInferenceModel(ABC):

    @abstractmethod
    def build_model(self, configs ):
        pass
    
    @abstractmethod
    def load_reid_model(self, model, path ):
        pass
     
    @abstractmethod
    def extract_features(self, model, patches ):
        pass


    def __init__(self, config_file, weights_path, device='cuda', batch_size=8):
        super(BaseInferenceModel, self).__init__()
        
        if str(device) not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be 'cpu' or 'cuda'. Got {device}")
    
        self.device=device
        self.batch_size=batch_size

        self.cfg = self.setup_cfg(config_file, weights_path)

        self.model = self.build_model(self.cfg)
        self.load_reid_model(self.model,weights_path)
        self.model.eval()

        print(f"Currently running on {device_count()} gpus.")
        self.model = self.model if device_count() <2 else nn.DataParallel(self.model)
        self.model = self.model.cuda()


        self.model = self.model.to(device=self.device)
        self.model.eval()
        if self.device != 'cpu':
            self.model = self.model.half()
        
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST
        #self.firstFrameProcessed=False

    
    def setup_cfg(self, config_file, opts):
        cfg = get_cfg()
        #print(f"CFG == {cfg}",flush=True)
        #cfg.merge_from_file(config_file)
        #cfg.merge_from_list(opts)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()
    
        return cfg

    def normalization_experiments(self, frame_patches, frame_number ):

        # Update variable on the first frame, whether it's 0 or 1
        mom_val=1.0 if frame_number==1 else 0.1
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats() #init mean=0.0, var=1.0, num_batches_tracked=0.0
                module.momentum = mom_val

        with torch.no_grad():
            # feed through network
            patches = torch.cat(frame_patches, dim=0)
            self.model(patches, forAdaptation=True)

        # set to eval mode again
        self.model.eval()

    def inference(self, image, detections, frame_number):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
            # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.to(device=self.device).half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))
        # features = np.zeros((0, 768))

        #self.normalization_experiments( batch_patches, frame_number )

        for patches in batch_patches:

            # Run model
            patches_ = torch.clone(patches)
            pred = self.extract_features(self.model, patches) # questo sostituisce !!! #pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

                        plt.figure()
                        plt.imshow(patch_np)
                        plt.show()

            features = np.vstack((features, feat))

        return features

   
