#TODO: Import your dependencies.
import json
import logging
import os
import sys
#import CrossEntropyLoss
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import ImageFile


import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, loss_criterion):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_criterion(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct,           len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))


    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''



def train(model, train_loader, test_loader, loss_criterion, optimizer, epochs):
    for epoch in range(1, epochs +1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()))
        test(model, test_loader, loss_criterion)
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''


def net():
    model = models.resnet50(pretrained =True)
    num_features = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(num_features, 133))

    return model

    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''


def create_train_loader(data_dir, batch_size):
    #data_dir = "dogbread"

    logger.info("Get train data loader")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    training_dir = os.path.join(data_dir, "train/")
    train_transform = transforms.Compose([transforms.RandomResizedCrop((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    train_data = datasets.ImageFolder(training_dir, transform = train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle=True)

    return train_loader

def create_test_loader(data_dir, test_batch_size):

    logger.info("Get test data loader")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    test_dir = os.path.join(data_dir , "test/")
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()])

    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= test_batch_size)

    return test_loader


    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''


def main(args):

    '''
    TODO: Initialize a model by calling the net function
    '''

    model=net()


    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    train_loader = create_train_loader(args.data_dir, args.batch_size)
    test_loader = create_test_loader(args.data_dir, args.test_batch_size)
    model=train(model, train_loader, test_loader, loss_criterion, optimizer, args.epochs)

    '''
    TODO: Test the model to see its accuracy
    '''

    test(model, test_loader, loss_criterion)


    '''
    TODO: Save the trained model
    '''
    logger.info("saving the model.")
    path = os.path.join(args.model_dir, "resnet50_model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    """
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    """

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])


    args=parser.parse_args()

    main(args)
