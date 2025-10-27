#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import cv2
import torchvision.transforms.functional as TF
from tqdm import tqdm
import tqdm as notebook_tqdm
tqdm.pandas()


# In[2]:


import os
import sys

# Add root path to sys.path so imports work
sys.path.append(os.path.abspath(os.path.join('..')))


# In[ ]:


pip install ipywidgets

pip install --upgrade jupyter ipywidgets


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("subirbiswas19/skin-disease-dataset")

print("Path to dataset files:", path)


# In[4]:


for root, dirs, files in os.walk(path):
    print("Folder:", root)


# In[5]:


train_data_path = os.path.join(path,"skin-disease-datasaet","train_set")
test_data_path = os.path.join(path,"skin-disease-datasaet","test_set")


# In[6]:


train_list_dir = os.listdir(train_data_path)
test_list_dir = os.listdir(test_data_path)


# In[7]:


train_list_dir


# In[8]:


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


# In[9]:


data_train = []

for disease_name in train_list_dir:
    disease_folder_path = os.path.join(train_data_path, disease_name)
    train_disease_pic_names = os.listdir(disease_folder_path)

    for pic_name in train_disease_pic_names:
        pic_path = os.path.join(disease_folder_path, pic_name)
        data_train.append({"image_path": pic_path, "label": disease_name})


df_train = pd.DataFrame(data_train)
df_train["label_values"] = df_train["label"].map(label_mapping)


# In[10]:


df_train


# In[11]:


data_test = []

for disease_name in test_list_dir:
    disease_folder_path = os.path.join(test_data_path, disease_name)
    test_disease_pic_names = os.listdir(disease_folder_path)

    for pic_name in test_disease_pic_names:
        pic_path = os.path.join(disease_folder_path, pic_name)
        data_test.append({"image_path": pic_path, "label": disease_name})

df_test = pd.DataFrame(data_test)
df_test["label_values"] = df_test["label"].map(label_mapping)


# In[12]:


df_test


# In[13]:


label_counts =  df_train["label"].value_counts().reset_index()
label_counts.columns = ["Disease", "count"]
print(label_counts)
plt.figure(figsize=(18,8))
ax = sns.barplot(x=  "Disease", y = "count", data = label_counts)
plt.title("class distribution")
ax.bar_label(ax.containers[0])
plt.show()


# In[14]:


from PIL import Image

image = Image.open(df_train["image_path"].iloc[0])
image_array = np.array(image)

display(image)


# In[15]:


import cv2


# In[16]:


gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
display(Image.fromarray(gray))


# In[17]:


import numpy as np


# In[18]:


norm_image = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
plt.imshow(norm_image)


# In[19]:


from albumentations import (
    Resize, Compose, Normalize, HorizontalFlip,
    GridDistortion, RandomBrightnessContrast
)

# For training
train_agumentation = Compose([
    Resize(128, 128),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# For minority class augmentation (optional)
unbalanced_transform = Compose([
    Resize(128, 128),
    HorizontalFlip(p=0.6),
    GridDistortion(p=0.6),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# For validation/test (no random ops!)
test_agumentation = Compose([
    Resize(128, 128),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# In[20]:


label_counts


# In[21]:


x_train = []
y_train = []

x_test = []
y_test = []

# Count current samples for each class in df_train
class_counts = df_train["label_values"].value_counts().to_dict()

# Find max samples among all classes (target balance count)
target_count = max(class_counts.values())

# Track how many extra we’ve added for each class
counter = {i: 0 for i in class_counts.keys()}

# First pass — add original images (augmented once)
for idx, row in df_train.iterrows():
    img_path = row["image_path"]
    label = row["label_values"]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_aug = train_agumentation(image=img)['image']
    x_train.append(img_aug.flatten())
    y_train.append(label)

# Second pass — oversample until balanced
for label in class_counts.keys():
    while counter[label] + class_counts[label] < target_count:
        # Randomly pick an image from this class
        sample_row = df_train[df_train["label_values"] == label].sample(1).iloc[0]
        img = cv2.imread(sample_row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_aug = unbalanced_transform(image=img)['image']  # stronger augmentations
        x_train.append(img_aug.flatten())
        y_train.append(label)
        counter[label] += 1

# Test set stays the same
for idx, row in df_test.iterrows():
    img_path = row["image_path"]
    label = row["label_values"]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = test_agumentation(image=img)['image']
    x_test.append(img.flatten())
    y_test.append(label)


# In[22]:


len(x_train), len(y_train), len(x_test), len(y_test)


# In[23]:


x_train ,  x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


# In[24]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[25]:


import numpy as np

unique_vals, counts = np.unique(y_train, return_counts=True)

print("Class distribution:")
for val, count in zip(unique_vals, counts):
    print(f"Class {val}: {count} samples")


# In[27]:


from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


lgbm = LGBMClassifier(n_estimators=50, max_depth=5, force_col_wise=True)

lgbm.fit(x_train, y_train)
preds = lgbm.predict(x_test)


# In[32]:


print(classification_report(y_test, preds))


# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[34]:


dtc = DecisionTreeClassifier()
dtc_param_grids = {
    "criterion" :["gini", "entropy", "log_loss"]
}
dtc_GSCV = GridSearchCV(dtc,  dtc_param_grids, cv = 10, scoring = "accuracy", verbose = 3)
dtc_GSCV.fit(x_train, y_train)
print(f"best params {dtc_GSCV.best_params_}")
print(f"best score : {dtc_GSCV.best_score_}")


# In[332]:


print(classification_report(y_test, dtc_pred))


# In[334]:


print(confusion_matrix(y_test, dtc_pred))


# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)


# In[37]:


print(classification_report(y_test, rfc_pred))


# In[38]:


print(confusion_matrix(y_test, rfc_pred))


# In[30]:


from sklearn.neighbors import KNeighborsClassifier

param_grid = { "n_neighbors" : range(1, 21), "weights": ['uniform', 'distance'], "algorithm" :['auto', 'ball_tree', 'kd_tree', 'brute'],
             "p" :[1, 2]}
knn = KNeighborsClassifier()

GSCV_scores = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy", verbose= 3)
GSCV_scores.fit(x_train, y_train)


# In[31]:


GSCV_scores.best_params_


# In[32]:


GSCV_scores.best_score_


# In[39]:


from sklearn.svm import SVC


# In[40]:


svc = SVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)


# In[41]:


print(classification_report(y_test, svc_pred))


# In[42]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()


# In[43]:


GNB.fit(x_train, y_train)
GNB_pred = GNB.predict(x_test)


# In[44]:


print(classification_report(y_test, GNB_pred))


# In[45]:


print(confusion_matrix(y_test, GNB_pred))


# In[29]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# 1️⃣ Convert numpy → torch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# In[30]:


# 2️⃣ Reshape to CNN format: (batch, channels, height, width)
x_train_tensor = x_train_tensor.view(-1, 3, 128, 128)
x_test_tensor = x_test_tensor.view(-1, 3, 128, 128)

# 3️⃣ Normalize pixel values (assuming 0–255 range)
x_train_tensor = x_train_tensor / 255.0
x_test_tensor = x_test_tensor / 255.0


# In[31]:


# 4️⃣ Create datasets and dataloaders
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[32]:


import torchvision


# In[33]:


dataiter = iter(train_loader)
images, labels = next(dataiter)
plt.imshow(np.transpose(torchvision.utils.make_grid(
    images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0)))
plt.axis('off')
plt.show()


# Simple CNN

# In[37]:


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16 * 16, 512),  # updated for 128x128 input
            torch.nn.ReLU(),
            torch.nn.Linear(512, 8)  # number of classes
        )


    def forward(self, x):
        return self.model(x)


# In[38]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = CNN().to(device)

num_epochs = 50
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# In[39]:


train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end=' ')
    train_loss = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss_list.append(train_loss / len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")


# In[40]:


plt.plot(range(1, num_epochs + 1), train_loss_list)
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")
plt.show()


# In[41]:


test_acc = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        y_true = labels.to(device)
        outputs = model(images)
        _, y_pred = torch.max(outputs.data, 1)
        test_acc += (y_pred == y_true).sum().item()

print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")


# In[ ]:




