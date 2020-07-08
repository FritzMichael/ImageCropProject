import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datasets import ImageDataSet, CroppedOutImageDataSet
from architectures import SimpleCNN
import gc

target_device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')
#target_device = 'cpu'

model = SimpleCNN(n_in_channels=4)
model.to(target_device)
print(target_device)

# Creating Dataset
root_dir = 'data'
dataset = CroppedOutImageDataSet(ImageDataSet(root_dir))

# Splitting dataset into train and test sets
trainSet, testSet = torch.utils.data.random_split(dataset, [int(len(dataset)*(4/5)), len(dataset) - int(len(dataset)*(4/5))])

# Creating Dataloaders
trainloader = DataLoader(dataset,batch_size=1,shuffle=True, num_workers = 0)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


mse = torch.nn.MSELoss()

def calcLoss(outputs, inputs, targets):
    target_mask = torch.squeeze(inputs)[1].to(dtype=torch.bool)
    pred = torch.squeeze(outputs)[target_mask]
    return mse(pred, targets.reshape(-1,))

def visualize_result(outputs, inputs, targets, update):
    fig, axs = plt.subplots(1,3,figsize=(30,5))
    axs[0].imshow(outputs, cmap= 'Greys_r')
    axs[0].set_title('result')
    axs[1].imshow(np.squeeze(inputs)[0,...], cmap= 'Greys_r')
    axs[1].set_title('cropped image')
    axs[2].imshow(np.squeeze(targets))
    axs[2].set_title('target')
    plt.savefig(f'results/{update}.pdf')
    plt.close()

update = 0
model.train()
while update < 1e3:
    for data in trainloader:
        inputs, targets, ids = data
        inputs = inputs.to(target_device)
        targets = targets.to(target_device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Calculate loss
        loss = calcLoss(outputs, inputs, targets)
        loss.backward()
        optimizer.step()

        update += 1
        
        if (update%100 == 0):
            print(loss)
            visualize_result(np.squeeze(outputs.cpu().detach().numpy()),inputs.cpu().detach().numpy(), targets.cpu().detach().numpy(), update)


