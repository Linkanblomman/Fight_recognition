import torch
from torch import nn
import torch.optim as optim
import fight_recognition.resnet as resnet

def initialize_model(model_architecture=None, model_dataset=None, num_classes=2):
    model = None
    pretained_model = f"./fight_recognition/pretrained_models/r3d{model_architecture}_{model_dataset}_200ep.pth"
    checkpoint = torch.load(pretained_model)

    if model_architecture == 18 and model_dataset == 'K':
        """ Resnet18
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=700)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 18 and model_dataset == 'KM':
        """ Resnet18
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=1039)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    if model_architecture == 34 and model_dataset == 'K':
        """ Resnet34
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=700)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 34 and model_dataset == 'KM':
        """ Resnet34
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=1039)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 50 and model_dataset == 'K':
        """ Resnet50
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=700)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 50 and model_dataset == 'KM':
        """ Resnet50
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=1039)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 101 and model_dataset == 'K':
        """ Resnet101
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=700)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 101 and model_dataset == 'KM':
        """ Resnet101
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=1039)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 152 and model_dataset == 'K':
        """ Resnet152
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=700)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 152 and model_dataset == 'KM':
        """ Resnet152
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=1039)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 200 and model_dataset == 'K':
        """ Resnet200
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=700)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 200 and model_dataset == 'KM':
        """ Resnet200
        """
        model = resnet.generate_model(model_depth=model_architecture, shortcut_type='B', n_classes=1039)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
      print("Pre-trained model not found")

    print("Model:", "ResNet" + str(model_architecture))

    if(model_dataset == 'K'):
        print("Dataset: Kinetics-700\n")
    else:
        print("Dataset: Kinetics-700 and Moments in Time\n") 

    optimizer = optim.SGD(model.parameters(), lr=0) # default optimizer
    optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer from pretrained model

    print("Model parameters")
    # Get learning parameters
    for param_group in optimizer.param_groups:
        print("Learning rate:", param_group['lr'])
        print("Momentum:", param_group['momentum'])
        print("Weight_decay:", param_group['weight_decay'])

    return model