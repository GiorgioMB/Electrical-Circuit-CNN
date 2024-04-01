#%%
import torchvision as tv
import torch as t
import os 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import ssl
from PIL import Image 
ssl._create_default_https_context = ssl._create_unverified_context
classes = set()
boxes = {}
imgs = {}
for i in range(26):
    directory = f'drafter_{i}/csvs/'
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(directory + file)
            classes.update(df['object'].tolist())
            temp = df[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
            key_to_save = file.split('.')[0] 
            boxes[key_to_save] = t.from_numpy(temp)
            


for i in range(26):
    directory = f'drafter_{i}/images/'
    for file in os.listdir(directory):
        if file.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(directory + file) 
            resized_image = image.resize((2048, 2048))
            key_to_save = file.split('.')[0]
            imgs[key_to_save] = image
num_classes = len(classes) + 1
filtered = set(imgs.keys()).intersection(set(boxes.keys()))
imgs_filter = {k: imgs[k] for k in filtered}
boxes_filter = {k: boxes[k] for k in filtered}
class CustomDataset(Dataset):
    def __init__(self, images, boxes, transform=None):
        self.images = images
        self.boxes = boxes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = list(self.images.keys())[idx]
        print("Retrieving image: ", image_name)
        try:
            image = self.images[image_name]
            boxes = self.boxes[image_name]
        except Exception as e:
            print("Error: ", e, type(e))
            raise e

        if self.transform:
            image = self.transform(image)
        return image, boxes

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CustomDataset(images=imgs_filter, boxes=boxes_filter, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=1e-5, amsgrad = True)
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, targets in data_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}')

torch.save(model.state_dict(), 'faster_rcnn_model.pth')

# %%
