from torch.utils.data.dataset import Dataset as D
import os
import numpy as np
import random
import torch
import torchvision as tv
import cv2
from tqdm import tqdm


def find_excluded_seq(path):
    excluded_seq = []
    with open(path, 'r') as f:
        for line in f:
            excluded_seq.append(line.strip())
    return excluded_seq


def generate_indices(end, temp_size=5, temp_stride=1):
    indices = []
    indices.append([0, 0, 0, 0, 1, 2, 3])
    indices.append([0, 0, 0, 1, 2, 3, 4])
    indices.append([0, 0, 1, 2, 3, 4, 5])
    # indices.append([0, 0, 0, 1, 2])
    # indices.append([0, 0, 1, 2, 3])
    for i in range(0, end - temp_size + 2, temp_stride):
        ind = np.arange(i, temp_size + i)
        indices.append(ind)
    indices.append([end - 5, end - 4, end - 3, end - 2, end - 1,  end,  end])
    indices.append([end - 4, end - 3, end - 2, end - 1, end, end, end])
    indices.append([end - 3, end - 2, end - 1, end, end, end, end])
    # indices.append([end - 3, end - 2, end - 1, end, end])
    # indices.append([end - 2, end - 1, end, end, end])
    return indices

class Dataset(D):

    def __init__(self, path_compressed_data, path_gt_data, patch_size=128, temp_size=7, temp_stride=1,
                 is_test=False, max_video_number=1e4, max_seq_length=1e4, excluded_path=None,
                 return_name=False, is_cropped=True, start_from=0):
        super(Dataset, self).__init__()

        self.path_compressed_data = path_compressed_data
        self.path_gt_data = path_gt_data
        self.is_test = is_test
        self.temp_size = temp_size
        self.return_name = return_name
        self.is_cropped = is_cropped
        self.patch_size = patch_size

        comp_videos_folders = os.listdir(self.path_compressed_data)
        gt_videos_folders = os.listdir(self.path_gt_data)
        # length check
        videos = list(set(comp_videos_folders + gt_videos_folders))

        self.data = {}  # key -> video name, elements = list of gt and compressed frames
        self.indices = []

        print('> Loading video sequences...')
        # iterate through gt videos
        videos.sort()
        if excluded_path is not None:
            excluded_seq = find_excluded_seq(excluded_path)
        for video_number, video in tqdm(enumerate(videos, 1), total=len(videos)):
            if excluded_path is not None and video in excluded_seq:
                video_number -= 1
                continue
            compressed_path = path_compressed_data + video
            frames = os.listdir(compressed_path)
            frames.sort()
            frames = frames[start_from:]
            frame_number = 1
            for frame_number, frame in enumerate(frames, 1):
                if video not in self.data.keys():
                    self.data[video] = {}
                if is_cropped:
                    crops = os.listdir(os.path.join(compressed_path, frame))
                    crops.sort()
                self.data[video][frame_number - 1] = {'frames': []}
                if is_cropped:
                    for crop in crops:
                        self.data[video][frame_number - 1]['frames'].append(os.path.join(video, frame, crop))
                else:
                    self.data[video][frame_number - 1]['frames'] = os.path.join(video, frame)
                if frame_number >= max_seq_length:
                    break
            for indices in generate_indices(frame_number - 1, temp_size, temp_stride):
                if is_cropped:
                    if self.is_test:
                        for crop_number in range(len(crops)):
                            self.indices.append([video, indices, crop_number])
                    else:
                        self.indices.append([video, indices, [crop_number for crop_number in range(len(crops))]])
                else:
                    self.indices.append([video, indices])
            if video_number >= max_video_number:
                break

        print('> Done. Loaded %d video sequences.' % len(self.data.keys()))

    def __getitem__(self, index):

        if self.is_cropped:
            video, indices, crop = self.indices[index]
            if not self.is_test:
                crop = random.choices(crop, k=1)[0]
            frame_names = [self.data[video][index]['frames'][crop] for index in indices]
        else:
            video, indices = self.indices[index]
            frame_names = [self.data[video][index]['frames'] for index in indices]

        compressed = []
        for index in range(self.temp_size):
            compressed.append(torch.from_numpy(np.float32(cv2.cvtColor(cv2.imread(os.path.join(self.path_compressed_data, frame_names[index])), cv2.COLOR_BGR2RGB)) / 255).permute(2,0,1))


        compressed_y = torch.cat([frame for frame in compressed], dim=0)
        gt_frame = torch.from_numpy(np.float32(cv2.cvtColor(cv2.imread(os.path.join(self.path_gt_data, frame_names[self.temp_size // 2])), cv2.COLOR_BGR2RGB)) / 255).permute(2, 0, 1)

        if not self.is_test:  # augment sequences if not test
            compressed_y, gt_frame = self.augment_seq(compressed_y, gt_frame, crop=True) #crop=not self.is_cropped)

        if self.return_name:
            return (compressed_y, gt_frame, frame_names[self.temp_size // 2])
        else:
            return (compressed_y, gt_frame)

    def mirror_image(self, im):
        return tv.transforms.functional.hflip(im)

    def flip_image(self, im):
        return tv.transforms.functional.vflip(im)

    def crop_image(self, im, x, y):
        return im[:, x:x + self.patch_size, y:y + self.patch_size]

    def augment_seq(self, compressed, gt, crop=False):
        # data augmentation
        flip = False
        mirror = False
        if random.random() > 0.5:
            flip = True
        if random.random() > 0.5:
            mirror = True

        # apply augmentation to both compressed and gt frames
        if crop:
            # crop param
            h = random.randint(0, compressed.shape[1] - self.patch_size)
            w = random.randint(0, compressed.shape[2] - self.patch_size)

            # apply augmentation to both compressed and gt frames
            compressed = self.crop_image(compressed, h, w)
            gt = self.crop_image(gt, h, w)
        if flip:
            compressed = self.flip_image(compressed)
            gt = self.flip_image(gt)
        if mirror:
            compressed = self.mirror_image(compressed)
            gt = self.mirror_image(gt)
        return (compressed, gt)

    def __len__(self):
        return len(self.indices)



