#%%
import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms.functional import to_tensor
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import ssl

# Disabling SSL verification to allow image loading from web if needed
ssl._create_default_https_context = ssl._create_unverified_context

class Rescale(object):
    """Rescale the image in a sample to a given size."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']
        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w))

        # Scale the bounding boxes accordingly
        boxes = boxes * torch.tensor([new_w / w, new_h / h, new_w / w, new_h / h])

        return {'image': img, 'boxes': boxes}

transform = transforms.Compose([
    Rescale(256),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, images, boxes, transform=None):
        self.images = images
        self.boxes = boxes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = list(self.images.keys())[idx]
        image = self.images[image_name]
        box_data = self.boxes[image_name]

        if self.transform:
            image = self.transform(image)

        boxes = torch.as_tensor(box_data, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack([to_tensor(img) for img in images], 0)
    return images, targets


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

dataset = CustomDataset(imgs_filter, boxes_filter, transform=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

criterion = nn.SmoothL1Loss()

optimizer = optim.Adagrad(model.parameters(), lr=0.001)
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, targets in data_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(data_loader)}')

torch.save(model.state_dict(), 'faster_rcnn_model.pth')
