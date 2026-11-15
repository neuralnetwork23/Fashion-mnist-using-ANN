
# PROJECT : Fashion MNIST

import torch.nn as nn
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torchvision.transforms import v2

data = pd.read_csv("fmnist_small.csv")
duplicate = data.duplicated()

# training and testing the data

# all columns
X = data.iloc[:,1:].values
X_tensor = torch.tensor(X, dtype=torch.float32)
#print(X)

# only first column
y = data.iloc[:,0].values
y_tensor = torch.tensor(y, dtype= torch.long)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X_tensor,
                                                    y_tensor,
                                                    test_size=0.2,
                                                 random_state=42)
# viewing the image
"""

import matplotlib.pyplot as plt
image = X[0].reshape(28,28)
print(image.shape)
plt.imshow(image,cmap = "gray")
plt.show()
"""

# out of 6000 .... data selected for training for X

print(len(X_train))

# out of 6000 .... data selected for testing for X
print(len(X_test))

# Scaling the feature
"""
#since we are working on neural network so its better that the value we are working should
#be 0 to 1 for better performance its not mandatory but its efficient to use it.
"""
X_train = X_train / 255
X_test = X_test / 255

print("The X_train value after scaling",X_train)
print("The X_test value after scaling",X_test)

# creating custom DataSet Class
class customDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index],self.y[index]
dataset = customDataset(X,y)
print("The dataset is", dataset)

# creating the train dataset
train_data = customDataset(X_train,y_train)

# creating the test dataset
test_data = customDataset(X_test, y_test)

#create the dataloader
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)

# create the test dataloader
test_loader = DataLoader(test_data,batch_size=32,shuffle=True)

print("The train loader is", train_loader)
print("The test loader is", test_loader)
# Building the Model

y_true = y_test
y_pred = torch.argmax(X_test)
print("The y_pred is ", y_pred)
print("The y_true is",y_true)

# creating the loss function
loss_fn = nn.CrossEntropyLoss()
model = customDataset()
#creating the optimizer
optimizer = optim.SGD(model.parameters(),lr =0.01)
epochs = 100
for epoch in range(epochs):
    for batch_features, batch_labels in train_loader:
#Forward pass
     outputs = model(batch_features)
# calculating the loss
     loss = loss_fn(outputs,batch_labels)
#backward pass
     optimizer.zero_grad()
     loss.backward(retain_graph = True)
     optimizer.step()
     #print(f"The loss item is: {epoch+1}, {loss.item()} ")
#Accuracy of Model
def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/ len(y_pred)) * 100
    return acc
# eval model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    print("The outpur is", outputs)
    y_pred = torch.argmax(
        outputs, dim = 1
    )
    print("The y_pred is", y_pred)

    y_true = y_test
accuracy = accuracy_fn(y_pred, y_true)
print("The accuracy is",accuracy)

# To check for My Image
from PIL import Image
image = Image.open("my_image.png")
image = image.convert("L")
image = image.resize((28,28))
# Converting image to tensor
image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
my_image = image_transform(image)
my_image = 1 - my_image
print(my_image)
print("The my image size is", my_image.shape)

# flattening my_new image
my_image_flatten = torch.flatten(my_image, start_dim = 1)
print(my_image_flatten)
print(my_image_flatten.shape)

model.eval()
with torch.no_grad():
    outputs = model(my_image_flatten) # flatten fits because flatten has [1,784]
    print(outputs)
    print(outputs.shape)

# items
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
class_mapping = [(f"{label} -> {item}") for label, item in enumerate(classes)]
print(class_mapping)
prediction = torch.argmax(outputs,dim =1)
print("The image send by user is ",classes[prediction])
