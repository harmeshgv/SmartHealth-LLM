import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import timm
from tqdm import tqdm
import kornia.augmentation as K



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import LABELS_JSON, SKIN_TEST, SKIN_TRAIN




train_data_path = SKIN_TRAIN
test_data_path = SKIN_TEST

train_list_dir = os.listdir(train_data_path)
test_list_dir = os.listdir(test_data_path)

data_train = []

for disease_name in train_list_dir:
    disease_folder_path = os.path.join(train_data_path, disease_name)
    train_disease_pic_names = os.listdir(disease_folder_path)

    for pic_name in train_disease_pic_names:
        pic_path = os.path.join(disease_folder_path, pic_name)
        data_train.append({"image_path": pic_path, "label": disease_name})
        
df_train = pd.DataFrame(data_train)

data_test = []

for disease_name in test_list_dir:
    disease_folder_path = os.path.join(test_data_path, disease_name)
    test_disease_pic_names = os.listdir(disease_folder_path)

    for pic_name in test_disease_pic_names:
        pic_path = os.path.join(disease_folder_path, pic_name)
        data_test.append({"image_path": pic_path, "label": disease_name})
        
df_test = pd.DataFrame(data_test)

label_map = {
    "BA- cellulitis" : 0,
    'BA-impetigo':1,
 'FU-athlete-foot':2,
 'FU-nail-fungus':3,
 'FU-ringworm':4,
 'PA-cutaneous-larva-migrans':5,
 'VI-chickenpox':6,
 'VI-shingles':7
    
}

df_train["num_label"] = df_train["label"].map(label_map)

df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

df_test["num_label"] = df_test["label"].map(label_map)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

class SkinDataset(Dataset):
    def __init__(self, df, img_size=(224, 224)):
        self.df = df
        self.img_size = img_size
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(self.img_size)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['num_label']
        image = Image.open(img_path).convert("RGB")
        image = self.resize(image)
        image = self.to_tensor(image)
        
        # Convert label to tensor (if it's not already)
        label = torch.tensor(label, dtype=torch.long)  # Use torch.float32 for regression
        return image, label
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kornia_transform = nn.Sequential(
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406], device=device),
                std=torch.tensor([0.229, 0.224, 0.225], device=device))
).to(device)

dataset_train = SkinDataset(df_train)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

dataset_test = SkinDataset(df_test)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)
print("dataset done")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained EfficientNet
model = timm.create_model('efficientnet_b0', pretrained=True)

# Modify the final classification layer 
model.classifier = nn.Linear(model.classifier.in_features, 8)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader_train):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader_train):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader_test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "efficientnet_b0_skin_classifier_weights.pth")
