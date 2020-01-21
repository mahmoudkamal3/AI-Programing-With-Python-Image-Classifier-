import argparse
import json
import torch
from time import time, sleep
from torch import nn, optim
from torchvision import datasets, transforms, models 
from collections import OrderedDict

def get_input_args():
    parser = argparse.ArgumentParser(description = 'Image Classification-train.py')

    parser.add_argument('data_directory', action='store', 
                        help='data directory containing training and testing data')
    parser.add_argument('--save_dir', action='store', dest='save_directory', default='checkpoint.pth',
                        help='directory where to save checkpoints')
    parser.add_argument('--arch', action='store', dest='arch', default="vgg16",
                        type= str, help='pre-trained model')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', default= 0.001,
                        type= float, help= 'learning_rate')
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', default=[4096, 2048],
                        help='number of hidden layers')
    parser.add_argument('--dropout', action='store', dest='dropout', default= 0.5,
                        help='amount of dropout')    
    parser.add_argument('--output', action='store', dest='output', default= 102,
                        help='output size')
    parser.add_argument('--epochs', action='store', dest='epochs', default= 1,
                        type= int, help='number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='use gpu to train model')

    args = parser.parse_args()
    return args                
     

    
    
def load_data(data_dir):
    
    # folder path
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])]),

        'test_transforms': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])]),

        'valid_transforms': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])}
    # Load the datasets with ImageFolder
    image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform = data_transforms['train_transforms']),
                      'test_data': datasets.ImageFolder(test_dir, transform = data_transforms['test_transforms']),
                      'valid_data': datasets.ImageFolder(valid_dir, transform = data_transforms['valid_transforms'])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                   'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=20, shuffle=True),
                   'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32, shuffle=True)}                                                                                                                                        
    return image_datasets, dataloaders 

    
def device_power(use_gpu):
    if torch.cuda.is_available() and use_gpu == True:
        model.to('cuda')     
    else:
        model.to('cpu')
    
def load_json():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def load_arch(arch):
    ''' load pretrained CNN Network for image classification 
    '''
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    else:
        raise ValueError('please choose from "vgg16" or "alexnet"')
                         
    for param in model.parameters():
        param.requires_grad=False                 
    
    return model, input_size
                         
def build_model(model, input_size, learning_rate, hidden_units, use_gpu):
    cat_to_name = load_json()
    # hyper parameters
    input_size = model.classifier[0].in_features
    hidden_units = [4096, 2048]
    output_size = len(cat_to_name)
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units[0])),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                            ('relu2', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc3', nn.Linear(hidden_units[1], 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier                     
                
    # allows users to choose speed training  
    if torch.cuda.is_available() and use_gpu == True:
        model.to('cuda')     
    else:
        model.to('cpu')
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)                     
               
    return criterion, optimizer
                         
                         
def valid(model, dataloaders, criterion, use_gpu):
    test_loss = 0
    accurcy = 0
    for data in dataloaders['valid_loader']:
        images, labels = data
        if torch.cuda.is_available() and use_gpu == True:           
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            images, labels = images.to('cpu'), labels.to('cpu')
            
                         
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accurcy += equality.type(torch.FloatTensor).mean()
    return test_loss, accurcy                         
                         
                        
                         
def train_model(epochs, model, dataloaders, optimizer, criterion,  use_gpu):
    steps = 0
    print_every = 40

    for epoch in range(epochs):
        running_loss = 0
    
        for images, labels in dataloaders['train_loader']:
            model.train()
            steps += 1
            if torch.cuda.is_available() and use_gpu == True:
                images, labels = images.to('cuda'), labels.to('cuda')            
            else:
                images, labels = images.to('cpu'), labels.to('cpu')
            
                         
            optimizer.zero_grad()
        
            # forward and backward
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accurcy = valid(model, dataloaders, criterion, use_gpu)
                
                print("Epochs: {}/{}..".format(epoch+1, epochs),
                      "training loss: {:.3f}".format(running_loss / print_every),
                      "test loss: {:.3f}..".format(test_loss / len(dataloaders['valid_loader'])),
                      "accurcy: {:.3f}..".format(accurcy / len(dataloaders['valid_loader'])))
                running_loss = 0
            model.train()
        
    print("\nTraining process is complete !!!")
                     
                         
                         
def save_checkpoint(model, args, image_datasets, optimizer):
    
    # checkpoint 
    print("Our model: \n\n", model, "\n")
    print("the state dict keys: \n\n", model.state_dict().keys())     
                         
    # assignment train_data to model.class
    model.class_to_idx = image_datasets['train_data'].class_to_idx
                     
    # Save the checkpoint 
    checkpoint = {'epochs': args.epochs,
                  'arch': args.arch,
                  'hidden_units': [each for each in args.hidden_units],
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
            

    torch.save(checkpoint, args.save_directory) 
    print("the checkpoint has been saved, done!!!")
    
    
def main():
    start_time = time()
    
    args =  get_input_args()
    # preparing data
    image_datasets, dataloaders = load_data(args.data_directory)
    
    # build network
    use_gpu = args.gpu
    model, input_size = load_arch(args.arch)
    criterion, optimizer = build_model(model, input_size,args.learning_rate, args.hidden_units, use_gpu)
    
    # test accurcy and training network
    valid(model, dataloaders, criterion, use_gpu)
    train_model(args.epochs, model, dataloaders, optimizer, criterion, use_gpu)
    
    # save checkpoint
    #hidden_layers = args.hidden_units
    save_checkpoint(model, args, image_datasets, optimizer)
    
    end_time = time()
    #time
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60))) 
 
          
          
# Call to main function to run the program
if __name__ == "__main__":
    main()

    
            
     