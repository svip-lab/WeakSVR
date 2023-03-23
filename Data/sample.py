from torch.utils import data
import os
import time
import torch
import cv2
import numpy as np
from numpy.random import randint
from torch.utils import data
from torchvision import transforms as tf
from tqdm import tqdm
from PIL import Image

def frames_loader(video_dir_path, offsets,num_worker=2):
    class Sample_Dataset(data.Dataset):
        def __init__(self,
                     video_dir_path=None,
                     offsets=None,
                     augment=True):
            self.offsets = offsets
            self.video_dir_path = video_dir_path
            self.augment = augment
            if augment:
                self.aug_flip = True
                self.aug_crop = True
                self.aug_color = True
                self.aug_rot = True

        def __getitem__(self, index):
            frame_index = offsets[index]
            frame_path = os.path.join(self.video_dir_path, str(frame_index + 1) + '.jpg')
            frame = cv2.imread(frame_path)
            frame = Image.fromarray(frame[:, :, [2, 1, 0]])
            frame = self.preprocess(frame)
            return frame

        def preprocess(self, frame, apply_normalization=True):
            # Apply augmentation and normalization on a clip of frames
            # ipdb.set_trace()
            # Data augmentation on the frames
            means, stds = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms = []
            # transforms.append(tf.RandomHorizontalFlip(1))
            # transforms.append(tf.RandomResizedCrop((180, 320), (0.7, 1.0)))
            # transforms.append(tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))

            if not self.augment:
                # Flip
                if np.random.random() > 0.5 and self.aug_flip:
                    transforms.append(tf.RandomHorizontalFlip(1))

                # Random crop
                # transforms.append(tf.Resize([224,224]))
                if np.random.random() > 0.5 and self.aug_crop:
                    transforms.append(tf.RandomResizedCrop((180, 320), (0.7, 1.0)))

                # Color augmentation
                if np.random.random() > 0.5 and self.aug_color:
                    transforms.append(tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))

                # # Rotation
                # if np.random.random() > 0.5 and self.aug_rot:
                #     transforms.append(tf.RandomRotation(30))
            transforms.append(tf.ToTensor())
            # Normalization
            if apply_normalization:
                transforms.append(tf.Normalize(means, stds))
                transforms = tf.Compose(transforms)

                frame = transforms(frame)
            else:
                transforms = tf.Compose(transforms)
                frame = transforms(frame)

            t41 = time.time()
            # print('video_dir_path', 't41-t4:', t41 - t4)
            return frame

        def __len__(self):
            return len(self.offsets)

    sample_dataset = Sample_Dataset(video_dir_path=video_dir_path, offsets=offsets)



    bn = len(sample_dataset) // num_worker

    loaders = data.DataLoader(dataset=sample_dataset,
                              batch_size=bn,
                              shuffle=False,
                              num_workers=num_worker,
                              pin_memory=False)
    frames = None
    for i, datas in enumerate(loaders):
        if i == 0:
            frames = datas
        else:
            frames = torch.cat([frames, datas], dim=0)
    if frames is None:
        raise ValueError('frames is None')
    return frames.permute(1, 2, 3, 0)
