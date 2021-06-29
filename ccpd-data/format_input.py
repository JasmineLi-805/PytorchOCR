"""
Generates the TextLine file in the format below. 
* Tab separated.
* Only works for CCPD files

" 图像文件名                 图像标注信息 "

train_data/train_0001.jpg   简单可依赖
train_data/train_0002.jpg   用科技让复杂的世界更简单

"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

INPUT_DIR = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/val'
OUTPUT_DIR = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data'
ANNOTATION_FILENAME = 'val.txt'

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京",
             "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
             "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
       '8', '9', 'O']


def get_plate_label(filename):
    components = filename.split('-')
    # print(components)
    plate = components[4].split('_')
    label = PROVINCES[int(plate[0])]
    for i in range(1, len(plate)):
        label += ADS[int(plate[i])]
    return label


def get_lp_points(filename):
    components = filename.split('-')
    points = components[3].split('_')
    assert(len(points) == 4)
    res = []
    for p in points:
        res.append([int(pp) for pp in p.split('&')])
    # print(res)
    res = [res[2], res[3], res[0], res[1]] # order by [top left, bot left, bot right, top right]
    # print(res)
    return res

def get_blurriness(image_paths):
    """
    Get the blurriness of each image from the path name

    - image_paths (list): a list of str image paths.
    """
    blurriness = []
    for path in image_paths:
        blur = int(path[:-4].split('-')[-1])
        blurriness.append(blur)
    return blurriness

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    points = points.astype(np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    # dst_img_height, dst_img_width = dst_img.shape[0:2]
    # if dst_img_height * 1.0 / dst_img_width >= 1.5:
    #     dst_img = np.rot90(dst_img)
    return dst_img

########################
# Higher Level Methods #
########################

def generate_annotation_file_from_directory(in_dir_path, out_file_path):
    file_paths = os.listdir(in_dir_path)
    annotation = []
    for file in file_paths:
        path = os.path.join(in_dir_path, file)
        label = get_plate_label(file)
        annotation.append(path + '\t' + label)

    with open(out_file_path, 'w') as f:
        annotation = '\n'.join(annotation)
        f.write(annotation)

def crop_and_save_images(in_dir_path, out_dir_path):
    file_paths = os.listdir(in_dir_path)
    cropped_img_cnt = 0
    for file in file_paths:
        path = os.path.join(in_dir_path, file)
        points = np.array(get_lp_points(file))
        img = cv2.imread(path, 0)
        crp = get_rotate_crop_image(img, points)
        cv2.imwrite(os.path.join(out_dir_path, file), crp)

        cropped_img_cnt += 1
        if cropped_img_cnt % 100 == 0:
            print(f'processed {cropped_img_cnt}/{len(file_paths)} images...')


if __name__ == "__main__":
    # remove blurriness score < 15
    in_dir_path = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/val'
    out_file_path = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/val_blur15.txt'
    file_paths = os.listdir(in_dir_path)
    blur = get_blurriness(file_paths)
    clean_file = []
    for i in range(len(blur)):
        if blur[i] > 15:
            clean_file.append(file_paths[i])
    
    annotation = []
    for file in clean_file:
        path = os.path.join(in_dir_path, file)
        label = get_plate_label(file)
        annotation.append(os.path.join(in_dir_path, path) + '\t' + label)
    
    with open(out_file_path, 'w') as f:
        annotation = '\n'.join(annotation)
        f.write(annotation)

    print(f'uncleaned images = {len(file_paths)}')
    print(f'cleaned images = {len(clean_file)}')

