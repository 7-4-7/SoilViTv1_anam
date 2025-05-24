from torch.utils.data import Dataset
from PIL import Image
import os
from torch import nn
import torch
import torchvision
from torchvision.models import vit_b_16 as vitb_16
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

class TrainImageDataset(Dataset):
    """Custom dataset for test images"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))  # sorted for reproducibility
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_filenames[idx]  # return filename to trace later


class TestImageDataset(Dataset):
    def __init__(self, root_dir, image_ids, transform):
        self.root_dir = root_dir
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_id

class TinyVGG_FeatureExtractor(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int) -> None:
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

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.flatten(x)
        return x
    
    
class Evaluation:
    def __init__(self, model, device='cuda'):
        """
        Initializes the evaluation utility.

        Args:
            model (torch.nn.Module): The model used to extract features.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.device = device
        self.embeddings = None
        self.image_names = []

    def generate_embeddings(self, train_dataloader):
        """
        Extracts and stores embeddings from a dataloader.

        Args:
            dataloader (DataLoader): Loader yielding (image_tensor, filename)
        """
        self.model.eval()
        embeddings = []
        image_names = []

        # Calulate Embeddings From Train Dataloader
        with torch.no_grad():
            for batch_images, filenames in train_dataloader:
                batch_images = batch_images.to(self.device)
                features = self.model(batch_images)
                embeddings.append(features.cpu()) # IMPORTANT CPU
                image_names.extend(filenames)

        self.embeddings = torch.cat(embeddings, dim=0)
        self.image_names = image_names

    def binary_classify(self, test_loader, threshold=0.1, visualize=False, test_image_dir = None):
        """
        Predicts if test images are 'Soil' or 'Not Soil' using cosine similarity.

        Args:
            test_loader (DataLoader): Loader yielding (image_tensor, image_id)
            threshold (float): Cosine distance threshold
            visualize (bool): Show up to 5 predictions if True

        Returns:
            List[int]: Binary predictions (1 = Soil, 0 = Not Soil)
        """
        if self.embeddings is None: ######## RUN THIS FITST
            raise ValueError("Embeddings not found. Run `generate_embeddings()` first.")

        # Offers prediction on test dataloader only after embeddings has been generated

        predictions = []
        self.model.eval()
        soil_embeddings = self.embeddings.cpu().numpy()
        show_limit = 5
        shown = 0

        with torch.no_grad():
            for image_tensor, image_id in test_loader:
                image_tensor = image_tensor.to(self.device)
                test_embedding = self.model(image_tensor).cpu().numpy().reshape(1, -1)
                distances = cosine_distances(test_embedding, soil_embeddings)[0]
                min_distance = distances.min()
                prediction = 1 if min_distance <= threshold else 0 # 1 - > Soil
                predictions.append(prediction)

                if visualize and shown < show_limit: # Optional Viz
                    img_path = Path(test_image_dir) / image_id[0]
                    image = Image.open(img_path).convert('RGB')
                    plt.imshow(image)
                    plt.title(f'Prediction: {"Soil" if prediction == 1 else "Not Soil"} | Distance: {min_distance:.5f}')
                    plt.axis('off')
                    plt.show()
                    shown += 1

        return predictions
    
class ModelLoader():
    """Contains helper functions, class specifically designed for model loading activities."""
    
    def __init__(self, model_name, pretrained_model_path):
        self.model_name = model_name
        self.pretrained_model_path = pretrained_model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def pipeline_initiate(self, TinyVGG_FeatureExtractor=None):
        """Calls model preparation methods and returns the model."""
        if self.model_name == 'vit':
            self.preparemodel_vit()
        elif self.model_name == 'tinyvgg':
            if TinyVGG_FeatureExtractor is None:
                raise ValueError("TinyVGG_FeatureExtractor must be provided for tinyvgg model.")
            self.preparemodel_tinyvgg(TinyVGG_FeatureExtractor)
        else:
            raise Exception("Wrong model name passed.")
        return self.model

    def preparemodel_vit(self):
        """Loads ViT, removes classifier, disables gradients, sets eval mode."""
        try:
            vit = vitb_16(weights='IMAGENET1K_V1').to(self.device)
        except Exception as e:
            print(str(e))
            print('Error occurred while loading', self.model_name, 'architecture')
            return

        # Correction - Original pretrained model works best, my transfer learning based model compresss too much information in final layer [1000 -> 4]

        # try:
        #     model_state_dict = torch.load(self.pretrained_model_path, map_location=self.device)
        #     vit.load_state_dict(model_state_dict, strict=False)
        # except Exception as e:
        #     print(str(e))
        #     print("Error occurred while assigning pretrained model's state dict")

        for param in vit.parameters():
            param.requires_grad = False

        vit.eval()
        self.model = vit

    def preparemodel_tinyvgg(self, TinyVGG_FeatureExtractor):
        """Loads TinyVGG, removes classifier, disables gradients, sets eval mode."""
        try:
            tinyvgg = TinyVGG_FeatureExtractor(3, 10).to(self.device)
        except Exception as e:
            print(str(e))
            print('Error occurred while loading', self.model_name, 'architecture')
            return

        try:
            model_state_dict = torch.load(self.pretrained_model_path, map_location=self.device)
            tinyvgg.load_state_dict(model_state_dict, strict=False)
        except Exception as e:
            print(str(e))
            print("Error occurred while assigning pretrained model's state dict")

        for param in tinyvgg.parameters():
            param.requires_grad = False

        tinyvgg.eval()
        self.model = tinyvgg
