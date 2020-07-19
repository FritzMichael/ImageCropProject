import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datasets import ImageDataSet, CroppedOutImageDataSet, collate_Images, scaledImageDataSet
from architectures import SimpleCNN
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
import time
from torchvision import datasets
import os
import gc

target_device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')
#target_device = 'cpu'

# Creating a Tensorboard dashboard to view prograss
result_path = os.path.join('results', 'tensorboard',strftime("%Y-%m-%d,%H:%M:%S", gmtime()))
writer = SummaryWriter(log_dir=result_path)

model = SimpleCNN(n_in_channels=4, n_hidden_layers=5, kernel_size = 9)
model.to(target_device)
print(target_device)

# Creating Dataset
#root_dir = 'downscaled_data'
root_dir = 'data'

dataset = CroppedOutImageDataSet(scaledImageDataSet('downscaled_data.pkl'))

# Splitting dataset into train and test sets
trainSet, testSet = torch.utils.data.random_split(dataset, [int(len(dataset)*(4/5)), len(dataset) - int(len(dataset)*(4/5))])

# Creating Dataloaders
trainloader = DataLoader(dataset,batch_size=32, shuffle=True ,num_workers = 8, collate_fn=collate_Images)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = np.inf
torch.save(model, os.path.join(result_path, 'best_model.pt'))


mse = torch.nn.MSELoss()

def calcLoss(outputs, mask, targets):

    zeroTens = torch.zeros_like(targets, device=target_device)
    preds = torch.where(torch.unsqueeze(mask,1).to(dtype=bool), outputs, zeroTens)

    #for pred, output, cinput in zip(preds, outputs, mask):
    #    temp = output[0][cinput.to(dtype=torch.bool)]
    #    pred[0,0:temp.size()[0]] = temp
    #target_mask = torch.squeeze(inputs)[1].to(dtype=torch.bool)
    #pred = torch.squeeze(outputs)[target_mask]
    
    return mse(preds, targets)

def visualize_result(outputs, inputs, targets, update):
    fig, axs = plt.subplots(1,4,figsize=(30,5))
    axs[0].imshow(outputs, cmap= 'Greys_r')
    axs[0].set_title('result')
    axs[1].imshow(np.squeeze(inputs)[0,...], cmap= 'Greys_r')
    axs[1].set_title('cropped image')
    axs[2].imshow(np.squeeze(targets))
    axs[2].set_title('target')

    prediction = np.where(np.squeeze(inputs)[1,...],outputs,np.squeeze(inputs)[0,...])
    axs[3].imshow(prediction, cmap= 'Greys_r')
    axs[3].set_title('best prediction')
    plt.savefig(f'results/{update}.pdf')
    plt.close()

update = 0
epoch = 0
model.train()
while True:
    for data in trainloader:
        starttime = time.time()
        inputs, crop_dicts, targets, ids = data
        datatime = time.time()
        print(f'datatime: {datatime-starttime}')

        optimizer.zero_grad()

        loss = 0

        inputs = inputs.to(target_device)
        targets = targets.to(target_device)

        transfertime = time.time()
        print(f'transfertime: {transfertime-datatime}')
        
        outputs, mask = model(inputs, crop_dicts)

        forwardtime = time.time()
        print(f'forwardtime: {forwardtime-transfertime}')
        #for x,target in zip(inputs, targets):
        #    x = x.to(target_device)
        #    target = target.to(target_device)
        #    output = model(x.unsqueeze(0))
        #    loss += calcLoss(output, x, target)
        #loss /= len(inputs)

        #if loss < best_loss:
        #    best_loss = loss
        #    torch.save(model, os.path.join(result_path, 'best_model.pt'))

        # Calculate loss
        loss = calcLoss(outputs, mask, targets)

        losstime = time.time()
        print(f'losstime: {losstime-forwardtime}')

        loss.backward()

        backwardtime = time.time()
        print(f'backwardtime: {backwardtime-losstime}')

        optimizer.step()

        gradienttime = time.time()
        print(f'gradienttime: {gradienttime-backwardtime}')

        update += 1
        
        totaltime = time.time()-starttime

        print(f'totaltime: {totaltime}')
        print(update)
        if (update%10 == 0):
            writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)
            #writer.add_image(tag='current pred',img_tensor= np.squeeze(output.cpu().detach().numpy()), global_step=update, dataformats='HW')

        #if (update%100 == 0):
            #visualize_result(np.squeeze(output.cpu().detach().numpy()),x.cpu().detach().numpy(), target.cpu().detach().numpy(), update)
    
        if (update%1000 == 0):
            torch.save(model, os.path.join(result_path, 'latest_model.pt'))
    epoch += 1

