import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
import pandas as pd

# Downloads the MNIST dataset and divides it in two tasks
########################################################################################################################

def get(data_path):
    # Load dataset:
    full_set = pd.read_csv(data_path)
    # Split in two tasks
    df_circle = full_set[full_set['Using circle'] == 1]
    df_arrow = full_set[full_set['Using circle'] == 0]
    # Shuffle:
    df_circle = df_circle.sample(frac=1)
    df_arrow = df_arrow.sample(frac=1)
    # Split each of them into 8 subsets:
    df_circle = np.array_split(df_circle, 8)
    df_arrow = np.array_split(df_arrow, 8)

    data= {}
    task_outputs = []
    size = [1,29]
    actions = [['Vacuum cleaning', 'Mopping the floor', 'Carry warm food', 'Carry cold food', 'Carry drinks', 'Carry small objects (plates, toys)', 'Carry big objects (tables, chairs)', 'Cleaning (Picking up stuff)'], ['Vacuum cleaning', 'Mopping the floor', 'Carry warm food', 'Carry cold food', 'Carry drinks', 'Carry small objects (plates, toys)', 'Carry big objects (tables, chairs)', 'Starting conversation']]

    # Circle
    for i, action in enumerate(actions[0]):
        data[i] = {}
        data[i]['name'] = action
        data[i]['n_outputs'] = 2
        data[i]['train']= {'x': torch.tensor([df_circle[i].iloc[0:500,0:-8].values])[0],'y': torch.tensor([df_circle[i].iloc[0:500,-8+i].values])[0]}
        data[i]['test']= {'x': torch.tensor([df_circle[i].iloc[500:,0:-8].values])[0],'y': torch.tensor([df_circle[i].iloc[500:,-8+i].values])[0]}

    # Arrow
    for i, action in enumerate(actions[1]):
        data[i+8] = {}
        data[i+8]['name'] = action
        data[i+8]['n_outputs'] = 2
        data[i+8]['train']= {'x': torch.tensor([df_arrow[i].iloc[0:500,0:-8].values])[0],'y': torch.tensor([df_arrow[i].iloc[0:500,-8+i].values])[0]}
        data[i+8]['test']= {'x': torch.tensor([df_arrow[i].iloc[500:,0:-8].values])[0],'y': torch.tensor([df_arrow[i].iloc[500:,-8+i].values])[0]}

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']= data[t]['train']['x'].clone()
        data[t]['valid']['y']= data[t]['train']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        task_outputs.append((t, data[t]['n_outputs']))
        n += data[t]['n_outputs']
    data['ncla'] = n

    return data, task_outputs, size

########################################################################################################################
