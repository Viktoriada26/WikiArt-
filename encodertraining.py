import json
import torch
import tqdm
from torch.utils.data import DataLoader
from wikiart import WikiArtDataset, Autoencoder
from torch.optim import Adam
from torch.nn import MSELoss
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file", default="config.json")
args = parser.parse_args()
config = json.load(open(args.config))

def train_model(config):
    """
    Training of an autoencoder by using the WikiArt dataset

    Args:
        config: which includes the training directory, epochs and batch_size

    """
    modelfile = "autoencoder.pth"  # encoded model
    trainingdir = config["trainingdir"]
    device = torch.device("cuda:1")
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    traindataset = WikiArtDataset(trainingdir, device, is_train=False)
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    autoencoder = Autoencoder(latent_dim=128).to(device)
    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    criterion = MSELoss()

    for epoch in range(epochs):
        autoencoder.train()
        epoch_loss = 0
        for batch_id, (artimages, labels) in enumerate(tqdm.tqdm(loader)): 
            artimages = artimages.to(device)
            optimizer.zero_grad()

            encode, decode = autoencoder(artimages)
            loss = criterion(encode, artimages)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(loader)}")

    torch.save(autoencoder.state_dict(), modelfile)
    print(f"Model saved to {modelfile}")



train_model(config)
