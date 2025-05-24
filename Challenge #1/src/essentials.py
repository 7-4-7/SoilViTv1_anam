
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image


class SoilClassification(Dataset):

  def __init__(self, df:pd.DataFrame, root_dir : Path, transform : transforms):

    """Custom class for soil classfication. It creates the a dataset for training and testing our model"""

    self.df = df
    self.pairs = [] # Pairs : example =  [(image, class index), ... ]
      
    self.cls_to_idx = None # To be initiated later by mapping method
    self.idx_to_cls = None # To be initiated later by mapping method
      
    self.root_dir = root_dir # Root Directory : '/kaggle/input/soil-classification/soil_classification-2025'
    self.transform = transform # Custom transforms to be applied on images

    self.mapping() # Assigns cls_to_idx and idx_to_cls dictionaries

    self.create_pairs()  # Begins creating pairs

  def mapping(self):

    """ Creates mapping : cls -> idx and idx -> cls"""

    labels = sorted(list(self.df.soil_type.unique()))
    self.cls_to_idx = {label:idx for idx,label in enumerate(labels)}
    self.idx_to_cls = {idx:label for idx,label in enumerate(labels)}

  def create_pairs(self):

    """ Creates [(image1, label1), (image2, label2), ...] pairs """

    for _,row in self.df.iterrows():
      img_id = row['image_id']
      label_idx = self.cls_to_idx[row['soil_type']]
      img_path = self.root_dir / 'train' / img_id
      img = Image.open(img_path).convert("RGB")

      if self.transform: # NEEDED !!! NO FALLBACK FOR NOW !!!
        t_img = self.transform(img)
        self.pairs.append((t_img,label_idx))
          
      else: # Fallback added
          # Need to have transforms to accompany the incosistent image type and PIL to tensor conversion
          print("!!!! Stopped dataset creation transforms compulsory !!!!")

  def __getitem__(self, index): # COMPULSORY TO OVERLOAD ON INHERITING DATASET CLASS
        
    """Fetch Pair at index """  
    return self.pairs[index]

  def __len__(self): # COMPULSORY TO OVERLOAD ON INHERITING DATASET CLASS

    """Return length of pairs"""
    return len(self.pairs)

  def show_batch(self,n_explore : int = 2):

    """ Shows example """
      
    # Inclusion of this method is insipired from Fast AI's show batch method that let's us visualize samples of dataset
      
    imgs_to_show = random.sample(self.pairs,k = n_explore)# Randomly samples n_explore images the user wants to visualize
    
    plt.figure(figsize=(10, 5))
      
    for i, (img, label) in enumerate(imgs_to_show):  
        img = img.permute(1,2,0).numpy() # Converts the shape from Channel Height Width to Height Width Channel and the type to numpy for Matplotlib compatibility
        plt.subplot(1, n_explore, i + 1)
        plt.imshow(img) # Critical - Shows transformed/augmented image rather than the actual train images
        label_str = self.idx_to_cls[label]
        title_text = f"Label: {label_str}\nShape: {img.shape}"
        plt.title(title_text)
        plt.axis('off') # Hide axes for cleaner display
    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show()


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Assuming input image size is 224x224 → becomes 56x56 after pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 56 * 56, 1024),  # First added Linear layer
            nn.ReLU(),
            nn.Linear(1024, 256),                     # Second added Linear layer
            nn.ReLU(),
            nn.Linear(256, output_shape)              # Final output layer
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
