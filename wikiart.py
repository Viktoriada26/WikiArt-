import sys
import os
import torch
from torchvision.io import read_image
from sklearn.utils import resample
from collections import Counter
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm



class WikiArtImage:
    """
    Class for representing images from the WikiArt dataset.

    Attributes:
        imgdir(str): The directory where each image is found 
        filename(str): The filename of each image 
        label (str) : The class in which each image belongs to
        filedict (dict): Dictionary mapping filenames to WikiArtImage objects.
        indices (list): List of image indices in the dataset.
        loaded (bool): Indicates whether the image is loaded or not.
    """

    def __init__(self, imgdir, label, filename):
        """
        Initializes the WikiArt dataset
        
        Args:
            imgdir (str): Directory where each image is stored
            label (str): Class of each image
            filename (str): The filename of each image

        """
         
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self): 
        """
        Loads the image from the specified file path 

        Returns:
            torch.tensor: Image as a PyTorch tensor
        """
    
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True
        return self.image


class WikiArtDataset(Dataset):
    """
    A custom dataset class for loading the dataset
    
    Attributes:
        fixed_classes (list): List of fixed classes.
        filedict (dict): Dictionary mapping filenames to WikiArtImage objects.
        indices (list): List of image file names in the dataset.
        device (str): Device cpu or cuda
        is_train (bool): Indicates whether the dataset is for training or test.

    """
    
    fixed_classes = [
        'Early_Renaissance', 'Naive_Art_Primitivism', 'Impressionism', 'Mannerism_Late_Renaissance', 'Pointillism',
        'Baroque', 'Symbolism', 'Synthetic_Cubism', 'Action_painting', 'Art_Nouveau_Modern', 'Rococo',
        'Analytical_Cubism', 'Expressionism', 'Realism', 'Romanticism', 'High_Renaissance', 'Pop_Art',
        'Post_Impressionism', 'Contemporary_Realism', 'Color_Field_Painting', 'Minimalism', 'New_Realism',
        'Abstract_Expressionism', 'Northern_Renaissance', 'Ukiyo_e', 'Cubism', 'Fauvism'
    ]

    def __init__(self, imgdir, device="cpu", is_train=True, upsample=False):  

        """
        Initialize the WikiArtDataset.

        Args:
            imgdir (str): directory for images
            device (str): cpu or cuda
            is_train (bool): Indicates whether the dataset is for training or test
            upsample (bool): Indicates Whether to upsample the dataset (upsampling for training) 

        """
       
       
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        print("Gathering files for {}".format(imgdir))

        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]

            if arttype in self.fixed_classes:
                for art in artfiles:
                    filedict[art] = WikiArtImage(imgdir, arttype, art)
                    indices.append(art)

        print("...finished")

        self.classes = self.fixed_classes
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.device = device
        self.is_train = is_train  # in order to know when it's train or test dataset

        self.class_counts = self._calculate_class_distribution()

        if self.is_train and upsample:
            self.indices = self._upsample_indices()

        self.print_class_distribution()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]

        label = imgobj.label.strip()
        if label not in self.classes:
            print(f"Label '{label}' not found in classes list.")

        ilabel = self.classes.index(label)

        image = imgobj.get().to(self.device)

        return image, ilabel

    def _calculate_class_distribution(self):
        """
        Calculates the distribution of classes in the dataset.

        Returns:
            dict: A dictionary where the key is the label of each is class and the value is the count of each label

        """

        class_counts = {}
        for class_name in self.classes:
            class_counts[class_name] = 0

        for imgname in self.indices:
            label = self.filedict[imgname].label.strip()
            if label in class_counts:
                class_counts[label] += 1

        return class_counts

    def _upsample_indices(self):
        """
        Upsampling in order to balance the dataset

        Returns:
            upsampled_indices(list): List of the upsampled classes 
        """
        class_to_indices = {}
        for class_name in self.classes:
            class_to_indices[class_name] = []

        for imgname in self.indices:
            label = self.filedict[imgname].label.strip()
            if label in class_to_indices:
                class_to_indices[label].append(imgname)

        # maximum size
        max_class_count = 0
        for class_name in class_to_indices:
            class_size = len(class_to_indices[class_name])
            if class_size > max_class_count:
                max_class_count = class_size

        upsampled_indices = []
        for class_name in class_to_indices:
            class_indices = class_to_indices[class_name]
            if len(class_indices) > 0:
                upsampled = resample(
                    class_indices,
                    replace=True,
                    n_samples=max_class_count,
                    random_state=42
                )
                for idx in upsampled:
                    upsampled_indices.append(idx)

        return upsampled_indices

    def print_class_distribution(self):
        """
        Prints the number of samples per class
        """
        class_counts = {}
        for class_name in self.classes:
            class_counts[class_name] = 0

        for idx in self.indices:
            label = self.filedict[idx].label.strip()
            if label in class_counts:
                class_counts[label] += 1

        print("Class distribution (Number of samples per class):")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")


class WikiArtModel(nn.Module):
    """
    A CNN for image classification 
    
    Arguments:
        num_classes(int) : The number of classes. The default number is 27.


    """
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv2d = nn.Conv2d(3, 1, (4,4), padding=2)
        self.maxpool2d = nn.MaxPool2d((4,4), padding=2)
        self.flatten = nn.Flatten()
        self.batchnorm1d = nn.BatchNorm1d(105*105)
        self.linear1 = nn.Linear(105*105, 300)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        #print("convout {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("poolout {}".format(output.size()))        
        output = self.flatten(output)
        output = self.batchnorm1d(output)
        #print("poolout {}".format(output.size()))        
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.softmax(output)






class Autoencoder(nn.Module):
    """
    A CNN for encoding and decoding images
        encoder_cnn : feature extraction 
        encoder_fc : converts the features into a latent vector
        decoder_fc : expands the hidden representation back to features
        decoder_cnn : reconstruction of the image
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Flatten()  
            
        )
        
        self.flattened_size = 256 * 52 * 52  
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flattened_size, latent_dim),
            #nn.ReLU()
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.ReLU()
        )

        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(1, (256, 52, 52)),  
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  
        )

    def forward(self, x):
        features = self.encoder_cnn(x)  
        latent = self.encoder_fc(features)  

        # Decode
        decoded_flat = self.decoder_fc(latent)  
        reconstructed = self.decoder_cnn(decoded_flat)  

        return reconstructed, latent
