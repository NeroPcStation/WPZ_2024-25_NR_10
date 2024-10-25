import torch
import os
from CNN import CNN
from logger import Logger
import numpy as np
import random


def main(num_epochs=50):
    logger = Logger("CNN_logs.csv")
    labels = [[1,0,0,0,0],  # daisy
              [0,1,0,0,0],  # dandelion
              [0,0,1,0,0],  # rise
              [0,0,0,1,0],  # sunflower
              [0,0,0,0,1]]  # tulip
    
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset = torch.load("dataset_trenowania.pt")

    dataset = list(train_dataset)
    random.shuffle(dataset)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for image, label in dataset:
            image = image

            image.to(device)

            output = model(image)
            loss = criterion(output.flatten().cpu(), torch.tensor(labels[label], dtype=torch.float32))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        logger.log(epoch, round(epoch_loss/dataset.__len__(), 4))
        model.save("CNN")
            
if __name__ == '__main__':
    main()
