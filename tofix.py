for epoch in range(num_epochs):
    running_loss = 0.0
    for images, targets in data_loader:
        print(images)
        images = images.to(device)
        print('Loaded Images')
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print('Loaded y')
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}')

torch.save(model.state_dict(), 'faster_rcnn_model.pth')
