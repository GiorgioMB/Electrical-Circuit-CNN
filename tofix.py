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
