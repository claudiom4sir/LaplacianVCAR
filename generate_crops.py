import cv2
import os
from multiprocessing import Process, Queue
import random
import queue
import sys
import argparse


def central_crop(h, w, window_size=96, stride=24):
    new_h, new_w = h, w
    i = 0
    while (window_size+stride*i) < h:
        i += 1
    if h % (window_size + stride * i) != 0:
        new_h = window_size + stride * (i - 1)
    i = 0
    while (window_size + stride * i) < w:
        i += 1
    if w % (window_size + stride * i) != 0:
        new_w = window_size + stride * (i - 1)
    start_h = (h - new_h) // 2
    end_h = new_h + start_h
    start_w = (w - new_w) // 2
    end_w = new_w + start_w
    return start_h, end_h, start_w, end_w


def crop_frame(im, window_size, stride):
    sh = im.shape
    if str(sh) not in crop_size.keys():
        new_crop_size = central_crop(sh[0], sh[1])
        crop_size[str(sh)] = new_crop_size
    h_s, h_e, w_s, w_e = crop_size[str(sh)]
    im = im[h_s:h_e, w_s:w_e]
    h, w = im.shape[0], im.shape[1]
    crops = []
    for row in range(0, h - window_size + 1, stride):
        for col in range(0, w - window_size + 1, stride):
            crops.append(im[row:row+window_size, col:col+window_size])
    return crops


def generate_videos(thread_number, wsize, stride):
    while True:
        try:
            sequence = q.get(True,1)
            print(f'Thread {thread_number}: processing sequence {sequence}. Remaining: {q.qsize()}')
            frames = os.listdir(os.path.join(source, sequence))
            frames.sort()
            for frame in frames:
                frame_path = os.path.join(source, sequence, frame)
                frame_number = frame[:frame.find('.')]
                im = cv2.imread(frame_path)
                crops = crop_frame(im, window_size=wsize, stride=stride)
                save_path = os.path.join(target, sequence, frame_number)
                os.makedirs(save_path, exist_ok=True)
                for i, crop in enumerate(crops):
                    cv2.imwrite(os.path.join(save_path, str(i).zfill(4) + '.png'), crop)
        except queue.Empty:
            print(f'Thread {thread_number}: Finish.')
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--compressed_dir', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/test_18/raw/')
    parser.add_argument('--target_dir', type=str, default='./MFQEv2_cropped_raw/')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--num_processes', type=int, default=8)
    args = parser.parse_args()
    print('> Running with the following arguments:')
    for arg, value in vars(args).items():
        print(f'    {arg}: {value}')

    source = args.compressed_dir
    target = args.target_dir
    stride = args.stride
    wsize = args.patch_size

    crop_size = {}
    sequences = os.listdir(source)
    sequences.sort()
    random.shuffle(sequences)

    threads = [None] * args.num_processes
    q = Queue()
    for path in sequences:
        q.put(path)
    for core in range(len(threads)):
        threads[core] = Process(target = generate_videos, args=(core, wsize, stride))
        threads[core].start()
    for core in range(len(threads)):
        threads[core].join()
    print('Finished')


