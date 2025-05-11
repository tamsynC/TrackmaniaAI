import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DrivingDataset(Dataset):
    """
    Dataset for behavioral cloning from recorded Trackmania frames and key presses.
    Expects:
      - image_dir: folder with images like frame_00001.jpg
      - label_json_path: JSON file with a list of {"frame": ..., "keys": [...]}
    """
    def __init__(self, image_dir, label_json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor()
        ])

        # Load JSON labels
        with open(label_json_path, 'r') as f:
            raw_data = json.load(f)

        # Expecting list of dicts with 'frame' and 'keys'
        self.data = [(item["frame"], item["keys"]) for item in raw_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, keys = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Convert key labels to tensor
        label = torch.tensor(keys, dtype=torch.float32)

        return image, label

