
import re
import os
import pickle
import sys

from PIL import Image # for saving to diff folder
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from datetime import datetime
import matplotlib.image
import numpy as np
import random
import torch

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
exit()

'''
1. Run file from its location

This file should:
go 1 folder up:
    create workdir 
        Download into workdir the CIFAR10 batches and python.tar.gz file
    create images and labels dir
        unpack data into images->cifar->cifar10->by-image->(test, train, train+val, val) 

2. run main file from its location

#LINUX
python main.py \
    --dataset cifar10 \
    --labels data-local/labels/cifar10/1000_balanced_labels/00.txt \
    --arch cifar_shakeshake26 \
    --consistency 100.0 \
    --consistency-rampup 5 \
    --labeled-batch-size 62 \
    --epochs 180 \
    --lr-rampdown-epochs 210

#WINDOWS
orig
python main.py --dataset cifar10 --labels data-local/labels/cifar10/1000_balanced_labels/00.txt --arch cifar_shakeshake26 --consistency 100.0 --consistency-rampup 5 --labeled-batch-size 62 --batch-size 256 --epochs 180 --lr-rampdown-epochs 210

test
python main.py --dataset cifar10 --labels data-local/labels/cifar10/1000_balanced_labels/00.txt --arch cifar_shakeshake26 --consistency 100.0 --consistency-rampup 5 --labeled-batch-size 8 --batch-size 32 --epochs 180 --lr-rampdown-epochs 210

'''
startTime = datetime.now()
np.random.seed(1)
unpack = False #set to false after first time if experimenting on data

image_dir = os.path.abspath('../images/cifar/cifar10/by-image/')
#cwd = os.path.abspath(os.getcwd())
if not os.path.exists('../workdir'):
    os.makedirs('../workdir')
if not os.path.exists(os.path.join(image_dir, 'train')):
    os.makedirs(os.path.join(image_dir, 'train'))
if not os.path.exists(os.path.join(image_dir, 'val')):
    os.makedirs(os.path.join(image_dir, 'val'))

if not os.path.exists('../../data-local/labels/cifar10/1000_balanced_labels'):
    os.makedirs('../../data-local/labels/cifar10/1000_balanced_labels')

work_dir = os.path.abspath('../workdir')
test_dir = os.path.abspath(os.path.join(image_dir, 'test'))
trainval_dir = os.path.abspath(os.path.join(image_dir, 'train+val'))
train_dir = os.path.abspath(os.path.join(image_dir, 'train'))
val_dir = os.path.abspath(os.path.join(image_dir, 'val'))
label_dir = os.path.abspath('../../data-local/labels/cifar10/1000_balanced_labels')

cifar10 = CIFAR10(work_dir, download=True)

def load_file(file_name):
    with open(os.path.join(work_dir, cifar10.base_folder, file_name), 'rb') as meta_f:
        return pickle.load(meta_f, encoding="latin1")

def unpack_data_file(source_file_name, target_dir, start_idx):
    print("Unpacking {} to {}".format(source_file_name, target_dir))
    data = load_file(source_file_name)
    for idx, (image_data, label_idx) in tqdm(enumerate(zip(data['data'], data['labels'])), total=len(data['data'])):
        subdir = os.path.join(target_dir, label_names[label_idx])
        name = "{}_{}.png".format(start_idx + idx, label_names[label_idx])
        os.makedirs(subdir, exist_ok=True)
        image = np.moveaxis(image_data.reshape(3, 32, 32), 0, 2)
        matplotlib.image.imsave(os.path.join(subdir, name), image)
    return len(data['data'])

label_names = load_file('batches.meta')['label_names']
#print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))

if unpack:
    start_idx = 0
    for source_file_path, _ in cifar10.test_list:
        start_idx += unpack_data_file(source_file_path, test_dir, start_idx)
    print("test len", start_idx)
    start_idx = 0
    for source_file_path, _ in cifar10.train_list:
        start_idx += unpack_data_file(source_file_path, trainval_dir, start_idx)

###########################################################################################################
#Separate train+val (10f*5k images=50k images) into train (45k images) and val (10f*0.5k=5k images)
###########################################################################################################
labels = os.listdir(trainval_dir) 

c = 0
for label in labels: #10dirs: airplane, dog,...,car
    print("Separating {} from train+val into train and val".format(label))
    #create dirs and subdirs in val and train to mirror train+val
    if not os.path.exists(os.path.join(val_dir, label)):
        os.makedirs(os.path.join(val_dir, label))
    if not os.path.exists(os.path.join(train_dir, label)):
        os.makedirs(os.path.join(train_dir, label))

    trainval_subdir = os.path.join(trainval_dir, label)
    val_subdir = os.path.join(val_dir, label)
    train_subdir = os.path.join(train_dir, label)

    #save 1/10 of each label from train+val to val, rest to train
    trainval_items = np.array(os.listdir(trainval_subdir))
    k = int(len(trainval_items)/10) #5000/10=500

    val_items = np.random.choice(trainval_items, k, replace=False)
    train_items = np.setdiff1d(trainval_items, val_items, assume_unique=True)

    if len(os.listdir(val_subdir)) != val_items.shape[0]:
        for item in val_items:
            val_img = Image.open(os.path.join(trainval_subdir, item))   
            val_img.save(os.path.join(val_subdir, item))
    else:
        print("val {} already filled".format(label))

    if len(os.listdir(train_subdir)) != train_items.shape[0]:
        for item in train_items:

            train_img = Image.open(os.path.join(trainval_subdir, item))   
            train_img.save(os.path.join(train_subdir, item))
    else:
        print("train {} already filled".format(label))



#take 1000 labels, 100 for each 10 classes and use as labels for training data in /data-local/labels/cifar10/1000_balanced_labels/00.txt
# labels have form filename.png label, e.g. 7015_airplane.png airplane
label_data = np.empty((1000,2), dtype='object')

for c, label in enumerate(labels):

    train_items = np.array(os.listdir(os.path.join(train_dir, label)), dtype='object')
    label_data[(c*100):((c+1)*100), 0] = train_items[:100]
    label_data[(c*100):((c+1)*100), 1] = label

np.savetxt(os.path.join(label_dir, '00.txt'), label_data, fmt='%s', delimiter=' ')

totalTime = datetime.now() - startTime
print("Finished in {}".format(totalTime))
