import os
import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from kaggle import api

# Download data from Kaggle
if not os.path.exists("data") and os.path.exists("data/train") and os.path.exists("data/test"):
    api.dataset_download_files("imsparsh/flowers-dataset", path="data", unzip=True)


# Define directory paths
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Transform data for training 
transform = {
  'train': transforms.Compose([
    transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),       # Random rotation by 10 degrees
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize with mean and std
  ])
}

# Create dataset
train_dataset = datasets.ImageFolder(train_dir, transform['train'],None)




print(train_dataset) 
# Function to load and transform image
# Required because test data doesn't have classes (?)
def transform_image(image_path):
  # Open image with PIL
  img = Image.open(image_path)
 # print(img)

  img = transforms.Resize(256)(img) # Resize to 256x256
  img = transforms.CenterCrop(224)(img) # Crop to 224x224
  img = transforms.ToTensor()(img) # Convert to tensor
  img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)  # Normalize with ImageNet 
  return img.unsqueeze(0)  


test_images = []
test_labels = []  
for filename in os.listdir(test_dir): # Iterate through all files in the test directory
  # Only use jpg and png files
  if filename.endswith(".jpg") or filename.endswith(".png"):

    image_path = os.path.join(test_dir, filename)

    # Image transform
    image = transform_image(image_path)

    # Remove additional dimension
    image = torch.squeeze(image, 0)

    # Append image to list
    test_images.append(image)
test_labels.append(-1)

print (f'Class names: {train_dataset.classes}')

# Save data
torch.save(train_dataset, 'train_dataset.pt')
torch.save(test_images, 'test_images.pt')
torch.save(test_labels, 'test_labels.pt')

print("Finished preparing data")