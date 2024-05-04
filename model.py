#%%
import os
import csv
import xml.etree.ElementTree as ET
import json
import torchvision as tv
import torch as t
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import ssl
import gc
from PIL import Image
import shutil
import cv2
from torchvision.transforms import functional as F
import psutil
import warnings
warnings.filterwarnings('ignore') 
print("IMPORTED LIBRARIES")
ssl._create_default_https_context = ssl._create_unverified_context # Disabling SSL verification to allow image loading from web if needed
if not os.path.exists('dataset'):
    os.mkdir('dataset')
    os.mkdir('train')
    os.mkdir('test')
    os.mkdir('dataset/drafter_00')
    print("CREATED FOLDERS")
#%%
classes = set()
boxes = {}
imgs = {}

directory = f'.\\train\\'
for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)
            classes.update(df['object'].tolist())

label_mapping = {label: i for i, label in enumerate(classes, start = 1)}

num_classes = len(classes) + 1
print(classes)
gc.collect()
#%%
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets

class CustomDataset(Dataset):
    def __init__(self, folder, target_size=(1024, 1024)):
        self.folder = folder
        self.theight, self.twidth = target_size
        self.names = self.indexes()
    
    def indexes(self):
        names = []
        for filename in os.listdir(self.folder):
            if filename.endswith('.csv'):
                name = filename.split('.')[0]
                names.append(name)
        return names
    
    def __len__(self):
        count = 0
        for path in os.listdir(self.folder):
            if path.endswith('.csv'):
                count += 1
        return count

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder, self.names[idx] + '.jpg')
        boxes_path = os.path.join(self.folder, self.names[idx] + '.csv')
        df = pd.read_csv(boxes_path)
        temp = df[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        temp2 = [label_mapping[obj] for obj in df['object'].tolist()]
        boxes = {'boxes': t.from_numpy(temp), 'labels': t.tensor(temp2)}
        del temp, temp2
        gc.collect()
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        og_dims = torch.tensor([image.width, image.height, image.width, image.height])
        image = F.resize(image, (self.theight, self.twidth))

        scaled_boxes = (boxes['boxes'] / og_dims) * torch.tensor([self.twidth, self.theight, self.twidth, self.theight])
        # Validate and remove boxes with zero area
        valid_boxes, valid_indices = self.validate_boxes(scaled_boxes)

        image = F.to_tensor(image)
        boxes['boxes'] = valid_boxes.to(torch.float64)
        boxes['labels'] = torch.tensor([boxes['labels'][i] for i in valid_indices], dtype=torch.int64)
        
        del scaled_boxes
        gc.collect()
        return image, boxes

    def validate_boxes(self, boxes):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid_indices = (widths > 0) & (heights > 0)
        return boxes[valid_indices], valid_indices.nonzero(as_tuple=True)[0]

    
train_folder = f'.\\train\\'
test_folder = f'.\\test\\'
test_dataset = CustomDataset(test_folder)
train_dataset = CustomDataset(train_folder)
print("CREATED TRAIN DATASET")
print("CREATED TEST DATASET")

train_data_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, collate_fn = collate_fn, pin_memory = True)
test_data_loader = DataLoader(test_dataset, batch_size = 8, shuffle = True, collate_fn = collate_fn, pin_memory = True)
model = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=1e-5, amsgrad = True)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    model.to(device)
else:
    device = torch.device('cpu')
    model.to(device)

#%%
def print_gpu_memory():
    if torch.cuda.is_available():
        print("GPU Memory:")
        for i in range(torch.cuda.device_count()):
            t = torch.cuda.get_device_properties(i).total_memory
            r = torch.cuda.memory_reserved(i)
            a = torch.cuda.memory_allocated(i)
            f = r-a 
            print(f"  GPU {i}: Total = {t/(1024**3):.2f} GB, Reserved = {r/(1024**3):.2f} GB, Allocated = {a/(1024**3):.2f} GB, Free (within reserved) = {f/(1024**3):.2f} GB")
    else:
        print("No CUDA-capable device is detected")


def print_system_memory():
    memory = psutil.virtual_memory()
    total = memory.total / (1024**3)
    available = memory.available / (1024**3)
    print(f"System Memory: Total = {total:.2f} GB, Available = {available:.2f} GB")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()
print_gpu_memory()
print_system_memory()
#%%
print("TRAINING MODEL")
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    running_loss = None
    for images, targets in train_data_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.sum().backward()
        optimizer.step()
        if running_loss is None:
            running_loss = losses.sum().cpu().detach().numpy()
        else: 
            running_loss += losses.sum().cpu().detach().numpy()
        del losses, loss_dict, images, targets
        torch.cuda.empty_cache() ##This is to make sure it does not crash
        gc.collect()
    print_gpu_memory()
    print_system_memory()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data_loader)}')
print("MODEL TRAINED")
    
torch.save(model.state_dict(), 'faster_rcnn_model.pth')
# %%
