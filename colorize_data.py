from typing import Tuple

from skimage.color import rgb2lab
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
import PIL.Image
from torchvision.transforms.functional import resize


class ColorizeData(Dataset):
    def __init__(self, img_dir):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256, 256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256, 256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Resizing the transform
        self.t_transfom = T.Compose([T.Resize(size=(256, 256))])

        # Loading Images

        self.img_dir = img_dir  # Image Directory
        self.images = sorted(os.listdir(img_dir))

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        img_path = os.path.join(self.img_dir, self.images[index])  # Image path
        image = PIL.Image.open(img_path)    # Load each Image

        if image.mode == "RGB":  # Eliminate any grayscale images in the dataset

            input = self.input_transform(image)     # Input image
            gt = self.target_transform(image)   # Ground truth or Label

            # Conversion to LAB space

            in_lab = self.t_transfom(image)
            in_lab = np.asarray(in_lab)
            in_lab = rgb2lab(in_lab)
            in_lab = (in_lab + 128)/255  # Since values b/w -128,127
            in_ab = in_lab[:,:,1:3]  # Image in AB space
            in_ab = torch.from_numpy(in_ab.transpose((2,0,1))).float()


            return input, in_ab, gt
