import argparse
import torch
import json
import numpy as np
from torchvision import models
from collections import OrderedDict
from time import time, sleep
from torch import nn, optim
from PIL import Image
import pandas as pd

def get_input_args():
    parser = argparse.ArgumentParser(description = 'this scrept helps model to predicting ')
    
    parser.add_argument('--img_path', action = 'store', dest = 'img_path', default = './flowers/valid/29/image_04104.jpg')
    parser.add_argument('--checkpoint', action = 'store', dest = 'checkpoint_path', default = 'checkpoint.pth',
                        help = 'the checkpoint file') 
    parser.add_argument('--top_k', action = 'store', dest = 'top_k', default = 5,
                        help = 'predict how many categories to show')    
    parser.add_argument('--category_names', action = 'store', dest = 'category_names', default = 'cat_to_name',
                        help = 'the real names of mapping classes')
    parser.add_argument('--gpu', action='store_true', help='use gpu to train model')    

    args = parser.parse_args()
    return args 

    
def device_power(use_gpu):
    if args.gpu == 'gpu':
        model.to('cuda')                        
    else:
        model.to('cpu')                         
    return gpu 

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    # download pretrained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    else:
        raise ValueError('please choose from "vgg16" or "alexnet"')
    # freez parameteres
    for param in model.parameters():
        param.requires_grad = False

    hidden_units = [4096, 2048]
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units[0])),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                            ('relu2', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc3', nn.Linear(hidden_units[1], 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier      

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict( checkpoint['state_dict'])

    return model    
    
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    
    # Resize the images where shortest side is 256 pixels, keeping aspect ratio. 
    if image.width > image.height:
        factor = image.width/image.height
        image = image.resize(size=(int(round(factor*256,0)),256))
    else:
        factor = image.height/image.width
        image = image.resize(size=(256,(int(round(factor*256,0)))))
        
    # Crop out the center 224x224 portion of the image.
    image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))
    
    # Convert to numpy array
    np_image = np.array(image)
    np_image = np_image/255
    
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimension for PyTorch
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

    
def predict(image_path, model, topk=5, use_gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}

    # predict the class from an image file
    if torch.cuda.is_available() and use_gpu == True:
        model.to('cuda')     
    else:
        model.to('cpu')
        
    model.eval()
    img = process_image(image_path)        
    img = torch.from_numpy(img).float()
    img = torch.unsqueeze(img, dim=0)
    
    output = model.forward(img)
    preds = torch.exp(output).topk(topk)
    probs = preds[0][0].cpu().data.numpy()
    classes = preds[1][0].cpu().data.numpy()
    
    topk_labels = [idx_to_class[i] for i in classes]
    probs = probs.tolist()
    return probs, topk_labels

def load_json(category_names):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    start_time = time()    
    args =  get_input_args()
        
    model = load_checkpoint(args.checkpoint_path)  
    
    image_path = args.img_path    
    use_gpu = args.gpu
    probs, topk_labels = predict(image_path, model, args.top_k, use_gpu)
    
    cat_to_name = load_json(args.category_names)
    real_names = [cat_to_name[l] for l in cat_to_name]
    print(model)
    print(pd.DataFrame(probs, topk_labels))
    
    df = pd.DataFrame(list(zip(real_names, probs)))    
    print('predictions:', df)
    
    end_time = time()    
    #time
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60))) 
 
          
# Call to main function to run the program
if __name__ == "__main__":
    main()
    