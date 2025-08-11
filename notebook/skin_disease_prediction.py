#!/usr/bin/env python
# coding: utf-8

# In[72]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import display

import kornia.augmentation as K
from kornia.geometry.transform import Resize
from kornia.constants import Resample
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# In[73]:


import os
import sys

# Add root path to sys.path so imports work
sys.path.append(os.path.abspath(os.path.join('..')))


# In[74]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("subirbiswas19/skin-disease-dataset")

print("Path to dataset files:", path)

# In[75]:


for root, dirs, files in os.walk(path):
    print("Folder:", root)


# In[76]:


train_data_path = os.path.join(path,"skin-disease-datasaet","train_set")
test_data_path = os.path.join(path,"skin-disease-datasaet","test_set")

# In[77]:


train_list_dir = os.listdir(train_data_path)
test_list_dir = os.listdir(test_data_path)

# In[78]:


train_list_dir

# In[79]:


label_mapping = {
    'VI-chickenpox': 0,
    'BA-impetigo': 1,
    'PA-cutaneous-larva-migrans': 2,
    'FU-nail-fungus': 3,
    'BA- cellulitis': 4,
    'VI-shingles': 5,
    'FU-ringworm': 6,
    'FU-athlete-foot': 7
}


# In[80]:


data_train = []

for disease_name in train_list_dir:
    disease_folder_path = os.path.join(train_data_path, disease_name)
    train_disease_pic_names = os.listdir(disease_folder_path)

    for pic_name in train_disease_pic_names:
        pic_path = os.path.join(disease_folder_path, pic_name)
        data_train.append({"image_path": pic_path, "label": disease_name})

        
df_train = pd.DataFrame(data_train)
df_train["label_values"] = df_train["label"].map(label_mapping)


# In[81]:


df_train

# In[82]:


data_test = []

for disease_name in test_list_dir:
    disease_folder_path = os.path.join(test_data_path, disease_name)
    test_disease_pic_names = os.listdir(disease_folder_path)

    for pic_name in test_disease_pic_names:
        pic_path = os.path.join(disease_folder_path, pic_name)
        data_test.append({"image_path": pic_path, "label": disease_name})
        
df_test = pd.DataFrame(data_test)
df_test["label_values"] = df_test["label"].map(label_mapping)

# In[83]:


df_test

# In[84]:


label_counts =  df_train["label"].value_counts().reset_index()
label_counts.columns = ["Disease", "count"]
print(label_counts)
plt.figure(figsize=(18,8))
ax = sns.barplot(x=  "Disease", y = "count", data = label_counts)
plt.title("class distribution")
ax.bar_label(ax.containers[0])
plt.show()


# In[85]:


from PIL import Image

image = Image.open(df_train["image_path"].iloc[0])
image_array = np.array(image)

display(image)

# In[86]:


import cv2

# In[87]:


gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
display(Image.fromarray(gray))


# In[88]:


import numpy as np

# In[89]:


norm_image = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
plt.imshow(norm_image)

# In[90]:


from albumentations import Compose, Resize, HorizontalFlip, Normalize, GridDistortion, GaussNoise,  VerticalFlip, Rotate, ToTensorV2, RGBShift

train_transform = Compose([
    Resize(256, 256),
    HorizontalFlip(p=0.5),
    Rotate(p=0.1),
    VerticalFlip(p=0.5),
    GridDistortion(p=0.2),
    Normalize(),
    ToTensorV2()

])

unbalanced_transform = Compose([
        Resize(256, 256),
    HorizontalFlip(p=0.5),
    Rotate(p=0.3),
    GridDistortion(p=0.5),
        GaussNoise( p=0.5),

    VerticalFlip(p=0.6),
    Normalize(),
    ToTensorV2()


])

test_tranform = Compose([
    Resize(256, 256),
    Normalize(),
    ToTensorV2()
])


# In[91]:


from torch.utils.data import Dataset


class DataAgumentation(Dataset):
    def __init__(self, df : pd.DataFrame, train_transform, unbalanced_transform, test_tranform, specialClasses : list, unbalancedAgumentCount : int, mode: str):
        self.df = df.reset_index(drop = True)
        self.train_transform = train_transform
        self.unbalanced_transform = unbalanced_transform
        self.test_transform = test_tranform
        self.specialclasses = specialClasses
        self.unbalancedAgumentCount = unbalancedAgumentCount
        self.counter = {i:0 for i in specialClasses }
        self.mode = mode

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path =row["image_path"]
        label = row["label_values"]

        image = Image.open(img_path).convert("RGB")

        image = np.array(image)

      
        if self.mode == "train":
            if label in self.specialclasses and self.counter[label] <= self.unbalancedAgumentCount:
                image = self.unbalanced_transform(image=image)['image']
                self.counter[label]+=1
            image = self.train_transform(image = image)['image']

        else:
            image = self.test_transform(image = image)['image']  


        return image, label  






# In[92]:


from torch.utils.data import DataLoader

# In[93]:


label_counts

# In[94]:


train_dataset= DataAgumentation(df= df_train,  train_transform=train_transform, unbalanced_transform=unbalanced_transform, test_tranform=test_tranform, specialClasses=[6, 7], unbalancedAgumentCount=  50, mode= "train")
test_dataset= DataAgumentation(df= df_test,  train_transform=train_transform, unbalanced_transform=unbalanced_transform, test_tranform=test_tranform, specialClasses=[6, 7], unbalancedAgumentCount=  50, mode= "test")

# In[95]:


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# In[96]:


images, labels = next(iter(train_loader))
print(images.shape)   # e.g., torch.Size([32, 3, 224, 224])
print(labels.shape)   # e.g., torch.Size([32])


# In[10]:


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

# In[11]:


df_train["num_label"] = df_train["label"].map(label_map)

# In[12]:


df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)


# In[13]:


df_train

# In[14]:


df_test["num_label"] = df_test["label"].map(label_map)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)


# In[15]:


df_test

# In[ ]:


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch
import kornia.augmentation as K
import kornia.geometry.transform as KG
from tqdm import tqdm

class SkinDataset(Dataset):
    def __init__(self, df, img_size=(224, 224)):
        self.df = df
        self.img_size = img_size
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(self.img_size)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]
        label = self.df.iloc[idx]['num_label']
        image = Image.open(img_path).convert("RGB")
        image = self.resize(image)
        image = self.to_tensor(image)
        
        # Convert label to tensor (if it's not already)
        label = torch.tensor(label, dtype=torch.long)  # Use torch.float32 for regression
        return image, label

# In[17]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# In[18]:


kornia_transform = nn.Sequential(
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406], device=device),
                std=torch.tensor([0.229, 0.224, 0.225], device=device))
).to(device)

# In[19]:


dataset_train = SkinDataset(df_train)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

dataset_test = SkinDataset(df_test)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

# In[185]:


import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):  # Now supports 8 classes
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers (adjust input size if needed)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)  # Input after conv layers
        self.fc2 = nn.Linear(512, num_classes)    # Output = 8 classes
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (32, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (64, 28, 28)
        
        # Flatten for FC layers
        x = x.view(-1, 64 * 28 * 28)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output shape: (batch_size, 8)
        
        return x

# In[186]:


import torch.optim as optim
from torch.utils.data import DataLoader

# Initialize
model = SimpleCNN(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader_train):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader_train):.4f}")

# In[187]:


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader_test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get class index
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# In[38]:


print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
print(f"Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

# In[20]:


dataset_train = SkinDataset(df_train)
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

# In[21]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=8):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Adaptive pooling layer to handle varying input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=8):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# In[22]:


# Initialize
import torch.optim as optim

model = ResNet18(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation code remains the same as before

# In[ ]:


import torch.optim as optim
from torch.utils.data import DataLoader

# Initialize
model = ResNet18(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader_train):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader_train):.4f}")

# In[1]:


import torch, gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


# In[24]:


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader_test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get class index
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# In[20]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm  # For EfficientNet models


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained EfficientNet
model = timm.create_model('efficientnet_b0', pretrained=True)

# Modify the final classification layer 
model.classifier = nn.Linear(model.classifier.in_features, 8)
model = model.to(device)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[23]:


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


# In[23]:


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


# In[28]:


torch.save(model.state_dict(), "efficientnet_b0_skin_classifier_weights.pth")


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd

# Count each label
label_counts = df_train["label"].value_counts().sort_index()

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(label_counts.index, label_counts.values, color='skyblue', edgecolor='black')

# Add count labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), ha='center', va='bottom')

plt.title("Label Distribution in Training Data")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[35]:


label_counts

# In[41]:


import torch, gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


# In[88]:


import torch
import timm
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def load_model(model_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 8)  # 8 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_tensor, class_names=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.unsqueeze(0).to(device)  # add batch dim

    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1).squeeze()
    predicted_idx = torch.argmax(probabilities).item()

    if class_names:
        return class_names[predicted_idx], probabilities.cpu().numpy()
    return predicted_idx, probabilities.cpu().numpy()


# In[89]:


from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
import kornia.augmentation as K

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224), device=None):
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor()
        ])

        self.kornia_norm = nn.Sequential(
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406], device=self.device),
                std=torch.tensor([0.229, 0.224, 0.225], device=self.device)
            )
        ).to(self.device)

    def preprocess(self, file) -> torch.Tensor:
        image = Image.open(file).convert("RGB")
        image = self.base_transform(image)
        image = image.to(self.device).unsqueeze(0)
        image = self.kornia_norm(image)
        return image.squeeze(0)  # Return [3, H, W]


# In[90]:


from image_preprocessing import ImagePreprocessor

processor = ImagePreprocessor(device=device)
model = load_model("efficientnet_b0_skin_classifier_weights.pth", device=device)

# In[91]:


image_tensor = processor.preprocess("skin-disease-datasaet/test_set/VI-chickenpox/14_VI-chickenpox (20).jpg")

# In[92]:


image_tensor

# In[93]:


label_map = {
    "BA-cellulitis": 0,
    "BA-impetigo": 1,
    "FU-athlete-foot": 2,
    "FU-nail-fungus": 3,
    "FU-ringworm": 4,
    "PA-cutaneous-larva-migrans": 5,
    "VI-chickenpox": 6,
    "VI-shingles": 7
}
idx_to_label = {v: k for k, v in label_map.items()}

# In[94]:


idx_to_label

# In[95]:



label, probs = predict(model, image_tensor, class_names=idx_to_label, device=device)

# In[96]:


print("Predicted label:", label)
print("Probabilities:", probs)

# In[104]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# In[105]:



# Paths to your data directories
train_dir = 'skin-disease-datasaet/train_set'
val_dir = 'skin-disease-datasaet/test_set'

# In[106]:




# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)


# In[107]:



# Load images from directories
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
)


# In[108]:



# Model setup using transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)  # 8 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# In[109]:




# Train the model
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# In[110]:


model.save('skin_disease_model_8_classes.h5')

# In[103]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import timm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paths to your data directories
train_dir = 'skin-disease-datasaet/train_set'
val_dir = 'skin-disease-datasaet/test_set'

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pretrained EfficientNet
model = timm.create_model('efficientnet_b0', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 8)  # 8 classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Validation / Test loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'skin_disease_efficientnet8.pth')
