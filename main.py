import torch
import torch.nn as nn 
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as tr
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import cv2
from PIL import Image
import torch.nn.functional as f
import matplotlib.pyplot as plt

device = torch.device("mps")
#i used mps but you can use cuda or cpu

root = "#path to images folder"

data = pd.read_csv("#path to csv bounding boxes")

scaler = MinMaxScaler()
data[['xmin', 'ymin', 'xmax', 'ymax']] = scaler.fit_transform(data[['xmin', 'ymin', 'xmax', 'ymax']])



transforms = tr.Compose(
    [
        tr.Resize((224,224)),
        tr.ToTensor(),
        tr.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5,))

    ]
)

class customdataset(Dataset):
    def __init__(self,df,transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,idx):
        x1 = torch.tensor(self.df.iloc[idx,:]["xmin"],dtype=torch.float)
        x2 = torch.tensor(self.df.iloc[idx,:]["xmax"],dtype=torch.float)
        y1 = torch.tensor(self.df.iloc[idx,:]["ymin"],dtype=torch.float)
        y2 = torch.tensor(self.df.iloc[idx,:]["ymax"],dtype=torch.float)

        image_list = glob.glob(root + self.df.iloc[idx,:]["image"])
        
        if image_list == str:
            image_list = [image_list]

        for img in image_list:
            image = Image.open(img)

        return self.transform(image),torch.tensor([x1,y1,x2,y2],dtype=torch.float)
    

train_data,test_data = train_test_split(data,test_size=0.2,random_state=42)


train_dataset = customdataset(train_data,transform=transforms)
test_dataset = customdataset(test_data,transform=transforms)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,64,3,1,1)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,256,3,1,1)
        self.conv5 = nn.Conv2d(256,512,3,1,1)

        self.fc1 = nn.Linear(25088,1024)
        self.fc2 = nn.Linear(1024,240)
        self.fc3 = nn.Linear(240,120)
        self.fc4 = nn.Linear(120,4)

    def forward(self,x):

        x = f.relu(f.max_pool2d(self.conv1(x),2,2))
        x = f.relu(f.max_pool2d(self.conv2(x),2,2))
        x = f.relu(f.max_pool2d(self.conv3(x),2,2))
        x = f.relu(f.max_pool2d(self.conv4(x),2,2))
        x = f.relu(f.max_pool2d(self.conv5(x),2,2))
        x = x.view(x.size(0),-1)
        
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    

model = CNN().to(device)

optim = torch.optim.Adam(model.parameters(),lr = 0.001)

criterion = nn.MSELoss()

num_epochs = 100

for epoch in range(num_epochs+1):
    for batch_idx,(images,labels) in enumerate(train_loader):
        images,labels = images.to(device),labels.to(device)

        outputs = model(images)

        optim.zero_grad()

        loss = criterion(outputs,labels)
        loss.backward()
        optim.step()

    print(f"Epoch: {epoch}; Loss: {loss.item()}")

torch.save(model,"model.pth")

model = torch.load("model.pth")
model.eval() 
with torch.inference_mode():
    for batch, (X,y) in enumerate(test_loader):
            
        X = X.to(device)
        y = y.to(device)
            
        test_pred = model(X)

idx = 5

test_pred_0 = test_pred[idx]
y_0 = y[idx]
X_0 = X[idx]

actual_corr = scaler.inverse_transform([y_0.cpu().numpy()])
predicted_corr = scaler.inverse_transform([test_pred_0.cpu().numpy()])

image_path = cv2.imread(glob.glob(root + data.iloc[idx,:]['image'])[0])
corr = actual_corr
cv2.rectangle(image_path, (int(corr[0][0]), int(corr[0][1])), (int(corr[0][2]), int(corr[0][3])),(255, 0, 0), 2)
plt.imshow(image_path)
plt.show()


img_path = "#image path to random image"
img = Image.open(img_path)

img_transformed = transforms(img)
img_transformed = img_transformed.unsqueeze(0)

with torch.inference_mode():
    model.eval()
    img_transformed = img_transformed.to(device)
    output = model(img_transformed)
    
    print("Scaled Output:", output)

predicted_corr_scaled = output[0].cpu().detach().numpy()  
predicted_corr = scaler.inverse_transform([predicted_corr_scaled])[0]

print("Unscaled Output:", predicted_corr)

image_path = cv2.imread(img_path)
cv2.rectangle(image_path, (int(predicted_corr[0]), int(predicted_corr[1])), (int(predicted_corr[2]), int(predicted_corr[3])), (0, 255, 0), 2)

cv2.imshow('Predicted Bounding Box', image_path)
cv2.waitKey(0)
cv2.destroyAllWindows()
