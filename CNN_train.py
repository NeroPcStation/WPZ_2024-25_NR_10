import torch
import os
from CNN import CNN
from logger import Logger
import numpy as np
import random
from PIL import ImageDraw, Image


def main(num_epochs=15):
    logger = Logger("CNN_logs.csv")
    
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset = torch.load("dataset_trenowania.pt", weights_only=False)

    dataset = list(train_dataset)
    random.shuffle(dataset)

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_guess = 0
        for image, label in dataset:
            image = image

            image.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output.cpu(), torch.tensor([label], dtype=torch.long))
            if output.argmax() == label:
                correct_guess += 1

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Acc: {round(correct_guess/dataset.__len__() * 100, 2)}%")
        logger.log(epoch, round(epoch_loss/dataset.__len__(), 4))
    torch.save(model, "CNN")


if __name__ == '__main__':
    main()