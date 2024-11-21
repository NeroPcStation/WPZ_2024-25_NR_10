import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CNN import CNN
from PIL import Image
import torch
import torchvision.transforms as transforms


def main(path):
      ResNet_label, CNN_label = GetLabels(path)
      print(f"ResNet: {ResNet_label}, CNN: {CNN_label}")

def GetLabels(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ResNet = torch.load(os.path.join(script_dir, "ResNet"), weights_only=False)
        # cnn = CNN()
        cnn = torch.load(os.path.join(script_dir, "CNN"), weights_only=False)


        ResNet_out = ResNet(ResNet_transform(path))
        CNN_out = cnn(CNN_transform(path))

        labels = ["daisy",  # daisy
                "dandelion",  # dandelion
                "rose",  # rose
                "sunflower",  # sunflower
                "tulip"]  # tulip

        _, predicted = torch.max(ResNet_out.data, 1)
        ResNet_label = labels[predicted]
        CNN_label = labels[torch.argmax(CNN_out)]
        return ResNet_label, CNN_label

def CNN_transform(path):
        transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(.5, .5)
        ])

        image = Image.open(path)
        return transform(image)

def ResNet_transform(path):
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

        image = Image.open(path)
        return transform(image).unsqueeze(0) 


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(f"Script name: {sys.argv[0]}")
        print(f"Dragged file path: {sys.argv[1]}")
        main(sys.argv[1])
