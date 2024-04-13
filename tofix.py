dataset = CustomDataset(images=imgs_filter, boxes=boxes_filter, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn =collate_fn)

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
        for imgs, targ in zip(images, targets):
            print(images)
            imgs = imgs.to(device)
            print('Loaded Images')
            targ = targ.to(device)
            print('Loaded y')
            optimizer.zero_grad()
            loss_dict = model(imgs, targ)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}')

torch.save(model.state_dict(), 'faster_rcnn_model.pth')
