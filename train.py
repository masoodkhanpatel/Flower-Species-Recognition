import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='flowers', help='Directory of the data')
    parser.add_argument('--save_dir', type=str, default='', help='Directory for saving the checkpoint')
    parser.add_argument('--arch', type=str, default = 'densenet161', help='The CNN model architecture for training')
    parser.add_argument('--learning_rate', type=float, default = 0.003, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default = 1000, help='Hidden units for the classifier')
    parser.add_argument('--epochs', type=int, default = 5 , help='Epochs')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if its available')
   
    return parser.parse_args()

def main():
    args = get_input_args()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Batch size
    batch_size = 64
    # For faster computation, setting num_workers
    num_workers = 4

    # Transforms for the training, validation, and testing sets
    data_transforms = {
        'train'      : transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),

        'valid'      : transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
        
    }

    # Loading the datasets with ImageFolder
    image_datasets = {
        'train'  : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid'  : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    # Using the image datasets and the trainforms in defining the dataloaders
    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size = batch_size, shuffle=True, num_workers = num_workers),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size = batch_size)
    }

    # Loading the pretrained model
    model = getattr(models, args.arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.classifier.in_features
    hidden_layer = args.hidden_units
    out_features = 102

    # Creating a custom classifier and attaching to the model
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(in_features, hidden_layer)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=.25)),
                            ('fc2', nn.Linear(hidden_layer, out_features)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    model.classifier = classifier

    device = torch.device("cuda" if args.gpu else "cpu")

    # Selecting the loss function and optimizer

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Moving the model to the device
    model.to(device);

    # Training the model
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 25
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                    f"Validation accuracy: { (accuracy/len(dataloaders['valid']))*100 :.3f}%")
                running_loss = 0
                model.train()

    # Save the checkpoint 
    model.class_to_idx = image_datasets['train'].class_to_idx


    checkpoint = {
                'batch_size': batch_size,
                'input_size': in_features,
                'output_size': out_features,
                'hidden_layers': [hidden_layer],
                'architecture' : args.arch,
                'lr' : args.learning_rate,
                'classifier' : classifier,
                'optimizer': optimizer.state_dict(),
                'class_to_idx': model.class_to_idx,    
                'state_dict': model.state_dict()
    }

    save_path = args.save_dir + '/checkpoint.pth'
    torch.save(checkpoint, save_path)
    print('Checkpoint Saved')

if __name__ == "__main__":
    main()

