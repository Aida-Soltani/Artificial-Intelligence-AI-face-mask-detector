import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import time as timer
import warnings

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer_01 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5),
                                       padding=(2, 2))  # nn.Conv2d(3, 10, 5)
        self.conv_layer_02 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(5, 5),
                                       padding=(2, 2))  # nn.Conv2d(10, 16, 5)
        self.conv_layer_03 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(5, 5),
                                       padding=(2, 2))  # nn.Conv2d(16, 10, 5)
        self.normal_layer_04 = nn.Linear(120, 84)
        self.normal_layer_05 = nn.Linear(84, 3)

        self.pool_layer = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # [n, 3, 32, 32]
        x = self.pool_layer(F.relu(self.conv_layer_01(x)))  # output dim= [n, 10, 16, 16]
        x = self.pool_layer(F.relu(self.conv_layer_02(x)))  # output dim= [n, 16, 8, 8]
        x = self.pool_layer(F.relu(self.conv_layer_03(x)))  # output dim= [n, 10, 4, 4]
        num_of_input_neurons = x.shape[1] * x.shape[2] * x.shape[3]  # get the size and number of images
        x = x.view(-1, num_of_input_neurons)
        x = F.relu(nn.Linear(num_of_input_neurons, 120)(x))
        x = F.relu(self.normal_layer_04(x))
        x = self.normal_layer_05(x)
        return x


def construct_model(mean, std, image_folder, shuffle_dataset):
    # normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std))
    ])

    normal_dataset = datasets.ImageFolder(image_folder, transform=transform)
    dataset_size = len(normal_dataset)
    kf = KFold(n_splits=10, shuffle= True)

    # Train
    learning_rate = 0.001
    num_epoch = 10

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    acc_list = []
    fold_exp_train = []
    fold_exp_test = []
    # print(total_step)
    for fold, (train_ids, test_ids) in enumerate(kf.split(normal_dataset)):
        # Print

        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold

        train_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=10, num_workers=2,
                                               sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=450, num_workers=2,
                                              sampler=test_sampler)

        print("Number of Train batches: ", len(train_loader))
        print("Number of Test batches: ", len(test_loader))
        total_step = len(train_loader)

        for epoch in range(num_epoch):
            for i, (images, labels) in enumerate(train_loader):
            #print(X_train.shape)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # back
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 40 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epoch, i + 1, total_step, loss.item(), (correct / total) * 100))


        model.eval()

        train_totall_acc= print_accuracy(model, train_loader, 'Train')
        test_totall_acc = print_accuracy(model, test_loader, 'Test')
        print_confusion_matrix(model, test_loader)
        fold_exp_train.append(train_totall_acc)
        fold_exp_test.append(test_totall_acc)

    return model, fold_exp_train, fold_exp_test

def print_accuracy(model, loader, label):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('\nAccuracy of the model on the {} images: {} %'.format(label, (correct / total) * 100))
        return (correct / total) * 100


def print_confusion_matrix(model, test_loader):
    images, test_labels = next(iter(test_loader))
    outputs1 = model(images)
    _, predicted2 = torch.max(outputs1.data, 1)
    print("\n", classification_report(test_labels, predicted2))
    print("Confusion Matrix:\n", confusion_matrix(test_labels, predicted2))


def calculate_stats(data_path):
    # Calculating mean and std of the whole dataset
    initial_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    initial_dataset = datasets.ImageFolder(data_path, transform=initial_transform)
    print('Number of images: ', len(initial_dataset))

    initial_dataloader = torch.utils.data.DataLoader(initial_dataset, batch_size=len(initial_dataset), num_workers=1)
    data = next(iter(initial_dataloader))
    return data[0].mean(), data[0].std()


if __name__ == '__main__':
    choose = input("You want to train(T) or load a model(L):")
    if choose == 'T':

        save_yes_no = input('Do you want to save the model after training, to be used later? (y/n)')
        if save_yes_no == 'y':
            path_saved_model = input('Please write the model name you wish to save: ')

        mean, std = calculate_stats('Dataset')
        print('mean: ', mean, ' std: ', std)
        tic = timer.time()
        model, fold_exp_train, fold_exp_test= construct_model(mean, std, 'Dataset', True)
        train_sum = 0
        for e in fold_exp_train:
            train_sum += e
        print("\n----- The Average Accuracy for train images:  ", train_sum/10, "%")
        test_sum = 0
        for e in fold_exp_test:
            test_sum += e
        print("\n----- The Average Accuracy for test images:  ", test_sum / 10, "%")

        toc = timer.time()
        print(('\nDuration = ', toc - tic))
        if save_yes_no == 'y' and len(path_saved_model) > 1:
            torch.save(model.state_dict(), path_saved_model)

    elif choose == 'L':
        load_path = input("Please write the model name you wish to load:")
        test_images_path = input('Path to the testing images:')
        model = CNN()
        model.load_state_dict(torch.load(load_path))
        model.eval()
        mean, std = calculate_stats(test_images_path)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((mean, mean, mean), (std, std, std))
        ])

        test_dataset = datasets.ImageFolder(test_images_path, transform=transform)
        testset_size = len(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testset_size)

        print_accuracy(model, test_loader, 'Test')
        print_confusion_matrix(model, test_loader)
    else:
        print('Valid options are (T)rain or (L)oad')