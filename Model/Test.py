import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from Model.Dataloader import ASLImageDataset
from Model.Net import Network
from torch.optim import Adam
from torch.autograd import Variable

train_transforms = transforms.Compose([
    transforms.RandomRotation(10),  # Slight rotations
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Lighting variations
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
])

batch_size = 10
number_of_labels = 26

train_set = ASLImageDataset(csv_file='../Dataset/sign_mnist_train/sign_mnist_train.csv',
                            root_dir='Dataset/sign_mnist_train/', transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0)

test_set = ASLImageDataset(csv_file='../Dataset/sign_mnist_test/sign_mnist_test.csv',
                           root_dir='Dataset/sign_mnist_test/', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z')


model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# Function to save the model
def saveModel():
    path = "./newestModel.pth"
    torch.save(model.state_dict(), path)


def testAccuracy(loader):
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for data in loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0
    test_accs = []
    train_accs = []

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0
                # Add test/train accuracy to list for later graphing
                train_accs.append(testAccuracy(train_loader))
                test_accs.append(testAccuracy(test_loader))

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy(test_loader)
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % accuracy)

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

    return train_accs, test_accs


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()[0]

    plt.imshow(npimg, cmap='gray')
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    images, labels = next(iter(test_loader))
    images = images.to(device)

    # show all images as one image grid
    imshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))


def testClasses():
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

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


# Let's build our model
epochs = 5
#train_accs, test_accs = train(epochs)
print('Finished Training')
"""plt.plot(range(2*epochs), train_accs, label="train")
plt.plot(range(2*epochs), test_accs, label="test")
plt.legend()
plt.show()"""

# Let's load the model we just created and test the accuracy per label
model = Network()
path = "99+AccModel.pth"
model.load_state_dict(torch.load(path))
model.eval()
print(testAccuracy(test_loader))
testClasses()
testBatch()
