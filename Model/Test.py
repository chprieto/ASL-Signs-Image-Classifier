import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from Model.Dataloader import ASLImageDataset
from Model.Net import Network
from torch.optim import Adam
from torch.autograd import Variable

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

batch_size = 10
number_of_labels = 26

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


model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)


def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    # Get device and move model to it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # Get ouputs for current weights, calculate loss and adjust weights
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Get statistics and print them during training
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        # Compute and print the accuracy for each epoch
        accuracy = testAccuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # Save model if it has greater accuracy than previous best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# Function to show images in batch
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()[0]

    plt.imshow(npimg, cmap='gray')
    plt.show()


# Test a batch of images, printing the correct and predicted values
def testBatch():
    # get batch of images from the test DataLoader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    images, labels = next(iter(test_loader))
    images = images.to(device)

    imshow(torchvision.utils.make_grid(images))

    # Get predicted labels and show both real and predicted values
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))


# Test the accuracy of each class individually
def testClasses():
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / (class_total[i] + 0.000001)))


# Train model
#train(5)
#print('Finished Training')

# Load model from file and print test statistics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network()
path = "myFirstModel.pth"
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

print(testAccuracy())
testClasses()
testBatch()
