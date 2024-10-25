import os
import datetime
import torch
import torchvision
import torchvision.transforms as transforms

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root='./data/train', 
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() - 1)

    # Load the ResNet50 model
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")

    # Parallelize training across multiple GPUs
    model = torch.nn.DataParallel(model)

    # Set the model to run on the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'Beginning training with {"GPU" if torch.cuda.is_available() else "CPU"}...')
    start_time = datetime.datetime.now()

    min_loss = 1

    # Train the model...
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

        if loss < min_loss:
            min_loss = loss
            torch.save(model, 'best')

        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    end_time = datetime.datetime.now()
    torch.save(model, 'last')
    print(f'Finished training, Loss: {loss.item():.4f}, Time: {end_time - start_time}')
    os.system("pause")

if __name__ == '__main__':
    main()