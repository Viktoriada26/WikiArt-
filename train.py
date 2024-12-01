import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtModel
import json
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")
args = parser.parse_args()
config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

print("Running...")


traindataset = WikiArtDataset(trainingdir, device, is_train=True, upsample=False)
#testingdataset = WikiArtDataset(testingdir, device)

print(traindataset.imgdir)
print("Training classes:", traindataset.classes)

the_image, the_label = traindataset[5]
print(the_image, the_image.size())




def train(epochs=3, batch_size=32, modelfile=None, device="cpu" ):
    """
    Training of the model 

    Args:
        epochs (int): Number of epochs to train the model. Default number is 3
        batch_size (int): Number of samples per batch. Default number is 32
        modelfile (str): Path to save the trained model after training
        device (str): cpu or cuda. Default cpu is used
    
    Returns:
        model: The WikiArt trained model.

    """
    
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss().to(device)
    
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(config["epochs"], config["batch_size"], modelfile=config["modelfile"], device=device)
