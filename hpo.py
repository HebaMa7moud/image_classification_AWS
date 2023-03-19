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
            loss = loss_criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == target.data)
        total_loss = test_loss / len(test_loader.dataset)
        total_acc = correct/ len(test_loader.dataset)
        
        logger.info(f"Testing Loss: {total_loss}")
        logger.info(f"Test Accuracy: {total_acc}")


    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''



def train(model, epochs,train_loader, validation_loader, loss_criterion, optimizer):
    epochs=epochs
    best_loss=1e6
    image_data= {'train':train_loader, 'valid':validation_loader}
    loss_count = 0

    for epoch in range(epochs):
        logger.info("Epoch".format(epoch))
        for img_dataset in ['train', 'valid']:
            if img_dataset== 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0
            corrects = 0
            for data, target in image_data[img_dataset]:
                output = model(data)
                loss = loss_criterion(output, target)
                if img_dataset=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                running_loss += loss.item() * data.size(0)
                _, preds = torch.max(output, 1)
                corrects += torch.sum(preds == target.data)

            epoch_loss = running_loss / len(image_data[img_dataset].dataset)
            epoch_acc = corrects / len(image_data[img_dataset].dataset)

            if img_dataset=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_count +=1

            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(img_dataset,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))
            print(epochs)
        if loss_count==1:
            break
        if epoch ==0:
            break
    return model



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

    model.fc = nn.Sequential(nn.Linear(2048, 128),
                             nn.ReLU(inplace=True),
                             nn.Linear(128, 133))

    return model

    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''


def create_train_loader(data_dir, batch_size):
    #data_dir = "dogbread"

    logger.info("Get train data loader")
    logger.info("Get validation data loader")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    training_dir = os.path.join(data_dir, "train/")
    valid_dir = os.path.join(data_dir, "valid/")

    train_transform = transforms.Compose([transforms.RandomResizedCrop((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    valid_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()])

    train_data = datasets.ImageFolder(training_dir, transform = train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle=True)

    validation_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader

def create_test_loader(data_dir, batch_size):

    logger.info("Get test data loader")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    test_dir = os.path.join(data_dir , "test/")
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()])

    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= batch_size, shuffle=True)

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

    train_loader, validation_loader = create_train_loader(args.data_dir, args.batch_size)
    test_loader = create_test_loader(args.data_dir, args.batch_size)
    model=train(model,args.epochs, train_loader, validation_loader, loss_criterion, optimizer)

    '''
    TODO: Test the model to see its accuracy
    '''

    test(model, test_loader, loss_criterion)


    '''
    TODO: Save the trained model
    '''
    logger.info("saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--epochs", type=int, default=5, metavar="E", help="learning rate (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )


    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    # Container environment
    
   
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])


    args=parser.parse_args()

    main(args)
