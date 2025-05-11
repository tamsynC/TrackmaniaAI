import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Constants
IMG_DIR = 'Trackmania_Coco_Segmentation/train'  # Replace with the actual image directory
ANN_FILE = 'Trackmania_Coco_Segmentation/train/_annotations.coco.json'

# cat_ids = self.COCO.getCatIds()
# print(f"Number of classes: {len(cat_ids)}, Category IDs: {cat_ids}")

NUM_CLASSES = 4 + 1  # Replace with the number of classes in your dataset
SAVE_PATH = 'unet_model.pth'  # Path to save the trained model

# Dataset class
class COCOSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.cat_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            # cat_id = ann['category_id']
            cat_id = self.cat_ids.index(ann['category_id'])  # Maps category_id to a 0-based index

            rle = self.coco.annToMask(ann)
            mask[rle > 0] = cat_id

        image = transforms.ToTensor()(image)

        return image, torch.tensor(mask, dtype=torch.long)

if __name__ == "__main__":
    train_dataset = COCOSegmentationDataset(IMG_DIR, ANN_FILE)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    )  #.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images, masks #.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")
