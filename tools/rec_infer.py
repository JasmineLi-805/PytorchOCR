# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib

from numpy.lib.type_check import _imag_dispatcher, imag

VERBOSE = True

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))

import numpy as np

sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter


class RecInfer:
    def __init__(self, model_path, batch_size=16):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
        self.converter = CTCLabelConverter(cfg['dataset']['alphabet'])
        self.batch_size = batch_size

    def predict(self, imgs):
        if VERBOSE:
            print(f'starting prediction...')
        # 预处理根据训练来
        if not isinstance(imgs,list):
            imgs = [imgs]
        imgs = [self.process.normalize_img(self.process.resize_with_specific_height(img)) for img in imgs]
        widths = np.array([img.shape[1] for img in imgs])
        idxs = np.argsort(widths)
        txts = []
        for idx in range(0, len(imgs), self.batch_size):
            if VERBOSE and idx % 1000 == 0:
                print(f'predicting {idx}th img out of {len(imgs)}')
            batch_idxs = idxs[idx:min(len(imgs), idx+self.batch_size)]
            batch_imgs = [self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1]) for idx in batch_idxs]
            batch_imgs = np.stack(batch_imgs)
            tensor = torch.from_numpy(batch_imgs.transpose([0,3, 1, 2])).float()
            tensor = tensor.to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                out = out.softmax(dim=2)
            out = out.cpu().numpy()
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        #按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        if VERBOSE:
            print(f'completed label prediction')
        return out_txts

def get_images_and_label(file_path):
    """
    Reads the images paths and corresponding annotations into a map

    - file_path (str): the path to the file with "<img path>\t<annotation>" on each line
    """
    if VERBOSE:
        print(f'reading img path and labels...')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        temp = []
        for line in lines:
            if line[-1] == '\n':
                temp.append(line[:-1])
            else:
                temp.append(line)
        lines = temp
        data = {
            'image_paths': [],
            'labels': []
        }
        for line in lines:
            line = line.split('\t')
            data['image_paths'].append(line[0])
            data['labels'].append(line[1])
        return data

def get_stats(pred, data, output_dir=None):
    def get_blurriness(image_paths):
        """
        Get the blurriness of each image from the path name

        - image_paths (list): a list of str image paths.
        """
        blurriness = []
        for path in image_paths:
            blur = int(path[:-4].split('-')[-1])
            # print(f'bluriness for {path}: {blur}')
            blurriness.append(blur)
        return blurriness

    if VERBOSE:
        print(f'start calculating accuracy...')
    stats = {}
    image_path = data['image_paths']
    images = data['images']
    bluriness = get_blurriness(image_path)
    targ = data['labels']
    pred = [p[0][0] for p in pred]
    assert(len(pred) == len(targ))
    correct_cnt = 0
    blur_to_correct = {}
    blur_to_miss = {}
    for i in range(0, len(pred)):
        # print(f'pred={pred[i]} - target={targ[i]}')
        if pred[i] == targ[i]:
            if bluriness[i] not in blur_to_correct:
                blur_to_correct[bluriness[i]] = 0
            blur_to_correct[bluriness[i]] += 1
            correct_cnt += 1
        else:
            if bluriness[i] not in blur_to_miss:
                blur_to_miss[bluriness[i]] = 0
            blur_to_miss[bluriness[i]] += 1
            if output_dir:
                cv2.imwrite(filename=os.path.join(output_dir, pred[i] + '.jpg'), img=images[i])
    stats['accuracy'] = 1.0 * correct_cnt / len(pred)
    stats['blur2miss'] = blur_to_miss
    stats['blur2hit'] = blur_to_correct
    return stats

def init_args():
    import argparse
    if VERBOSE:
        print(f'parsing arguments')
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    # parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    parser.add_argument('--data_info', required=True, type=str, help='path to the text file containing img path and annotation')
    parser.add_argument('--output_dir', required=False, type=str, help='directory to store incorrectly labelled imgs', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import cv2

    # args = init_args()
    # data = get_images_and_label(args.data_info)
    # # img = cv2.imread(args.img_path)
    # if VERBOSE:
    #     print(f'loading model...')
    # model = RecInfer(args.model_path)

    # if VERBOSE:
    #     print(f'loading images...')
    # data['images'] = [cv2.imread(img) for img in data['image_paths']]
    # labels = data['labels']

    # out = model.predict(data['images'])
    # stats = get_stats(out, data, output_dir=args.output_dir)
    # accuracy = stats['accuracy']
    # print(f'\ntotal # of images to predict: {len(labels)}; accuracy: {accuracy}')

    data_info = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/val.txt'
    model_path = '/Users/jasmineli/Desktop/PytorchOCR/output/CRNN/checkpoint/best.pth'

    print(f'Starting...')
    print(f'parsing image paths and annotations')
    data = get_images_and_label(data_info)
    print(f'loading model...')
    model = RecInfer(model_path)
    print(f'loading images...')
    data['images'] = [cv2.imread(img) for img in data['image_paths']]
    labels = data['labels']

    output_dir = '/Users/jasmineli/Desktop/PytorchOCR/eval'
    out = model.predict(data['images'])
    stats = get_stats(out, data, output_dir=output_dir)
    accuracy = stats['accuracy']
    print(f'\ntotal # of images to predict: {len(labels)}; accuracy: {accuracy}')