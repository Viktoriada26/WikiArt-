import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from wikiart import WikiArtDataset, Autoencoder
import argparse
import json
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file", default="config.json")
args = parser.parse_args()
config = json.load(open(args.config))

modelfile = "autoencoder.pth"
trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
#device = config["device"]
device = torch.device("cuda:2") 
epochs = config["epochs"]
batch_size = config["batch_size"]
num_clusters = 27

dataset = WikiArtDataset(testingdir, device, is_train=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

autoencoder = Autoencoder(latent_dim=128).to(device)
autoencoder.load_state_dict(torch.load(modelfile))
autoencoder.eval()


encoded_features = []
labels = []

print("Extracting encoded representations")
with torch.no_grad():
    for images, batch_labels in loader:
        images = images.to(device)
        decode, encode = autoencoder(images)
        encoded_features.append(encode.cpu().numpy())
        labels.extend(batch_labels.numpy())

encoded_features = np.concatenate(encoded_features, axis=0)
labels = np.array(labels)

# clustering using kmeans
print("Clustering representations...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(encoded_features)

# Reduce dimensions using TSNE
reduced_dimension = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(encoded_features)

# Visualization of the clusters
print("Creating cluster plot...")
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    reduced_dimension[:, 0],
    reduced_dimension[:, 1],
    c=clusters,
    cmap="tab20",
    #cmap="tab10",
    alpha=0.7
)


colors = [scatter.cmap(i / num_clusters) for i in range(num_clusters)]  # unique colors for clusters
legend_labels = [f"Cluster {i}" for i in range(num_clusters)]
custom_handles = [plt.Line2D([], [], marker='o', color=colors[i], linestyle='', markersize=8) for i in range(num_clusters)]



plt.legend(custom_handles, legend_labels, title="Clusters")



output_file = config.get("output_image", "clusters.png")
plt.savefig(output_file)
print(f"Cluster plot saved to {output_file}")
print(f"Number of unique clusters: {len(set(clusters))}")


clusterlabels = defaultdict(set)

# Mapping each cluster to the art styles it contains
for cluster, label in zip(clusters, labels):
    category_name = dataset.classes[label] 
    clusterlabels[cluster].add(category_name)


#for cluster, categories in clusterlabels.items():
    #print(f"Cluster {cluster}: {categories}")


