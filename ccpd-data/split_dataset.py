"""
Split the images in IMG_DIR into 70% training, 15% validation, and 15% testing
"""

import os
import random
import shutil


IMG_DIR = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/all'
TRAIN_DIR = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/train'
VAL_DIR = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/val'
TEST_DIR = '/Users/jasmineli/Desktop/PytorchOCR/ccpd-data/test'

images = os.listdir(IMG_DIR)
cnt = 0
for img in images:
    n = random.randint(1, 100)
    if n < 71:
        shutil.move(src=os.path.join(IMG_DIR, img), dst=TRAIN_DIR)
    elif  n < 86:
        shutil.move(src=os.path.join(IMG_DIR, img), dst=VAL_DIR)
    else:
        shutil.move(src=os.path.join(IMG_DIR, img), dst=TEST_DIR)
    
    cnt += 1
    if cnt % 500 == 0:
        print(f'moved {cnt}/{len(images)} images from {IMG_DIR}')