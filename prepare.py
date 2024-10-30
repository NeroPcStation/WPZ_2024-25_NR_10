import os
import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from kaggle import api

# Pobranie danych z Kaggle
if not os.path.exists("data") and os.path.exists("data/train") and os.path.exists("data/test"):
    api.dataset_download_files("imsparsh/flowers-dataset", path="data", unzip=True)


#definicja ścieżek do folderów
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Transofracje danych do trenowania 
transformacje_danych = {
  'train': transforms.Compose([
    transforms.RandomResizedCrop(224),  # Losowe przycięcie i powiększenie do 224x224
    transforms.RandomHorizontalFlip(),   # Losowe odwrócenie horyzontalne
    transforms.RandomRotation(10),       # Losowe obrócenie o 10 stopni
    transforms.Grayscale(num_output_channels=1),  # Konwersja do skali szarości z 1 kanałem
    transforms.ToTensor(),              # Konwersja do tensora
    transforms.Normalize([0.5], [0.5])  # Normalizacja z mean i std
  ])
}

#Stworzenie datasetu
dataset_trenowania = datasets.ImageFolder(train_dir, transformacje_danych['train'],None)




print(dataset_trenowania) 
# Funkcja do wczytania i przetworzenia obrazu
# Jest ona wymagana ponieważ dane testowe nie posiadają class
def wykonanie_transformacji(sciezka_obrazu):
  # otworzenie obtazku za pomocą PIL
  img = Image.open(sciezka_obrazu)
 # print(img)

  img = transforms.Resize(256)(img) #Skalowanie do 256x256
  img = transforms.CenterCrop(224)(img)# przyciecie do 224x224
  img = transforms.ToTensor()(img) # Konwersja do tensora
  img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)  # Normalizacja za pomocą ImageNet 
  return img.unsqueeze(0)  


testowe_zdjecia = []
testowe_znaczniki = []  
for filename in os.listdir(test_dir): # Przechodzimy przez wszystkie pliki w folderze testowym
  # Branie pod uwage tylko plików jpg i png
  if filename.endswith(".jpg") or filename.endswith(".png"):

    sciezka_obrazu = os.path.join(test_dir, filename)

    # Wykonanie transformacji na obrazie
    obraz = wykonanie_transformacji(sciezka_obrazu)

    # Usunięcie dodatkowej wymiaru
    obraz = torch.squeeze(obraz, 0)

    # Dodanie obrazu do listy
    testowe_zdjecia.append(obraz)
testowe_znaczniki.append(-1)

print (f'Nazwy klas: {dataset_trenowania.classes}')

#Zapisanie danych do plików
torch.save(dataset_trenowania, 'dataset_trenowania.pt')
torch.save(testowe_zdjecia, 'testowe_zdjecia.pt')
torch.save(testowe_znaczniki, 'testowe_znaczniki.pt')

print("Zakończono przygotowanie danych")