import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json


def get_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image' , type=str, default='*.jpg', help='Image for inference')
    parser.add_argument('--save' , type=str, default='checkpoint.pth', help='Path of your saved model')
    parser.add_argument('--category_names' , type=str, default='cat_to_name.json', help='file for mapping')
    parser.add_argument('--top_k' , type=int, default='5', help='How many top probabilites?')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if its available')
   
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Loading the image
    img = Image.open(image)
    
    # Preprocessing the image
    preprocess_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    return preprocess_transforms(img)

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Put the model in evaluation mode
    model.eval()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # Turn off gradients
    with torch.no_grad():
        image.to(device)
        output = model.forward(image)
        ps = torch.exp(output)
        
        # Calculate the topk probabilties, default is top 5
        top_p, top_class = ps.topk(topk)
        classes_probability = []
        
        for label in top_class.numpy()[0]:
            classes_probability.append(list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(label)])
    return top_p.numpy()[0], classes_probability

def main():
    args = get_input()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.save)
    print(model.classifier)
    print("Checkpoint Loaded.")

    image = args.image

    device = torch.device("cuda" if args.gpu else "cpu")
    # Moving the model to the device
    model.to(device)
    
    # Calculing the topk probabilities, default top 5
    top_p, top_classes = predict(image, model, args.top_k, device)

    # Empty list to store the predicted labels
    labels = []
    for classes in top_classes:
        labels.append(cat_to_name[classes])

    print("The top probabilities for the image are...")
    for each in range(args.top_k):
        print('{}. {} : {}'.format(each+1, labels[each], top_p[each]))

if __name__ == "__main__":
    main()


