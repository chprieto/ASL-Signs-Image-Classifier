import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from Model.Dataloader import (ASLImageDataset, ToTensor)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

batch_size = 4

train_set = ASLImageDataset(csv_file='../Dataset/sign_mnist_train/sign_mnist_train.csv',
                            root_dir='Dataset/sign_mnist_train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0)

test_set = ASLImageDataset(csv_file='../Dataset/sign_mnist_test/sign_mnist_test.csv',
                           root_dir='Dataset/sign_mnist_test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()[0]

    plt.imshow(npimg, cmap='gray')
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))