# Import Packages

import time

import numpy as np
import torch
import os

import PIL.Image
from matplotlib import pyplot as plt
from skimage.color import lab2rgb
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as T
from basic_model import Net
from colorize_data import ColorizeData as cd


# Collate to remove the b/w images from the dataset

def my_collate(batch):
    """Removes the None values from ColorizeData outputs"""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


# Convert lab to rgb

def lab_to_rgb(in_g, in_ab, path, name):
    """Converts the LAB images to RGB space
        in_g : Grayscale input image
        in_ab : Image in AB space
        path : Path to save the image
        """
    rgb_img = torch.cat((in_g, in_ab), 0).numpy()
    rgb_img = rgb_img.transpose((1, 2, 0))
    rgb_img[:, :, 0:1] = rgb_img[:, :, 0:1] * 100  # Converting the L space to [0,100]
    rgb_img[:, :, 1:3] = rgb_img[:, :, 1:3] * 255 - 128  # Converting the AB space
    rgb_img = lab2rgb(rgb_img.astype(np.float64))  # LAB to RGB conversion
    plt.imsave(arr=rgb_img, fname='{}{}'.format(path, name))  # Save the RGB image


# Dataloaders

dataset = cd(img_dir='landscape_images')
train_len = int(0.9 * len(dataset))  # Length of train dataloader
val_len = len(dataset) - train_len  # Length of validation  dataloader
torch.random.seed()

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True, collate_fn=my_collate, )
val_dataloader = DataLoader(val_dataset, batch_size=48, collate_fn=my_collate)


# Training

class Trainer:
    def __init__(self):
        self.criterion = torch.nn.MSELoss().cuda()

    def train(self, model, lr, optimizer, epochs):

        # Model

        criterion = torch.nn.MSELoss().cuda()  # Loss function to use

        # train loop

        model.train()
        train_running_loss = 0

        for i, data in enumerate(train_dataloader):
            inputs, inputs_ab, gts = data
            inputs = Variable(inputs).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, inputs_ab.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        train_loss = train_running_loss / len(train_dataloader)

        # Validation loop

        model.eval()
        val_running_loss = 0
        for i, data in enumerate(val_dataloader):

            inputs, inputs_ab, gts = data  # inputs_ab -> input image in AB space
            inputs = inputs.cuda()
            outputs = model(inputs)  # Outputs -> Predictions
            val_loss = self.criterion(outputs, inputs_ab.cuda())
            val_running_loss += val_loss.item()
            if epochs % 25 == 0:
                os.makedirs(f'outputs/{epochs}', exist_ok=True)
                for j in range(5):
                    lab_to_rgb(inputs[j].detach().cpu(), outputs[j].detach().cpu(),
                               path=f'outputs/{epochs}',
                               name='img-{}-epoch-{}.jpg'.format(i * val_dataloader.batch_size + j, epoch))

        val_loss = val_running_loss / len(val_dataloader)
        print(f'Training loss = {train_loss} ; Validation loss = {val_loss}')

        return train_loss, val_loss

    # Validate / Test Loop

    def validate(self, img_dir, index):

        """Run this from the inference script
           img_dir : Directory of input image
           index : Index of input image"""

        path = 'b2c.pt'     # Path of the model
        model.load_state_dict(torch.load(path))     # Load the model
        images = sorted(os.listdir(img_dir))
        t_transfom = T.Compose([T.ToTensor(), T.Resize(size=(256, 256))])   # Transforms
        for in_img in images:
            img_path = os.path.join(img_dir, images[index])  # Image path
            in_img = PIL.Image.open(img_path)
            in_img = t_transfom(in_img)
            in_img = torch.unsqueeze(in_img, dim=1)
            in_img = in_img.cuda()
            outputs = model(in_img)
            lab_to_rgb(in_img[0].detach().cpu(), outputs[0].detach().cpu(),
                       path=f'outputs/test',
                       name='img-{}-{}.jpg'.format(1, 1))       # Convert to RGB


# Training and validation

epochs = 50  # Num of epochs
train_losses = []  # List of training lossed
val_losses = []  # List of Validation losses
model = Net()  # Model
model = model.cuda()  # Load model to GPU
trainer = Trainer()  # Calling Trainer
lr = 1e-3  # Learning rate
run = False  # Training flag

if run == True:

    path = 'b2c.pt'  # Model Path

    for epoch in range(epochs):
        t0 = time.time()
        print(f'Epoch : {epoch + 1}')
        path = 'b2c.pt'
        if epoch >= 1:
            model.load_state_dict(torch.load(path))  # Load the previous model
            # Annealing
            if epoch % 20 == 0:
                lr *= 0.5  # Decaying learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        tr_loss, val_loss = trainer.train(model=model, lr=lr, optimizer=optimizer, epochs=epoch)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        torch.save(model.state_dict(), path)  # Save the model
        print('Model Saved!')
        print(f'Time taken for epoch {epoch + 1} is : ', time.time() - t0, ' seconds\n')

    # Plotting Loss and Accuracy curves

    plt.plot(train_losses, 'b')
    plt.plot(val_losses, 'r')
    plt.legend(["train", "val"])
    plt.title('LOSS VALUES')
    plt.show()
