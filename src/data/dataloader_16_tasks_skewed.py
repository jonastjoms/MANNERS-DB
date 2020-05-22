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
    # Give first 4 actions more data:
    # Circle
    df_circle_1 = df_circle.iloc[0:3652, :]
    df_circle_2 = df_circle.iloc[3652:, :]
    df_circle_1 = np.array_split(df_circle_1, 4)
    df_circle_2 = np.array_split(df_circle_2, 4)
    df_circle = [df_circle_1, df_circle_2]
    # Arrow
    df_arrow_1 = df_arrow.iloc[0:3480, :]
    df_arrow_2 = df_arrow.iloc[3480:, :]
    df_arrow_1 = np.array_split(df_arrow_1, 4)
    df_arrow_2 = np.array_split(df_arrow_2, 4)
    df_arrow = [df_arrow_1, df_arrow_2]

    data= {}
    task_outputs = []
    size = [1,29]
    actions = [['Vacuum cleaning', 'Mopping the floor', 'Carry warm food', 'Carry cold food', 'Carry drinks', 'Carry small objects (plates, toys)', 'Carry big objects (tables, chairs)', 'Cleaning (Picking up stuff)'], ['Vacuum cleaning', 'Mopping the floor', 'Carry warm food', 'Carry cold food', 'Carry drinks', 'Carry small objects (plates, toys)', 'Carry big objects (tables, chairs)', 'Starting conversation']]

    # Circle
    for i, action in enumerate(actions[0]):
        if i < 4:
            data[i] = {}
            data[i]['name'] = action
            data[i]['n_outputs'] = 2
            data[i]['train']= {'x': torch.tensor([df_circle[0][i].iloc[0:700,0:-8].values])[0],'y': torch.tensor([df_circle[0][i].iloc[0:700,-8+i].values])[0]}
            data[i]['test']= {'x': torch.tensor([df_circle[0][i].iloc[700:,0:-8].values])[0],'y': torch.tensor([df_circle[0][i].iloc[700:,-8+i].values])[0]}
        else:
            data[i] = {}
            data[i]['name'] = action
            data[i]['n_outputs'] = 2
            data[i]['train']= {'x': torch.tensor([df_circle[1][i-4].iloc[0:300,0:-8].values])[0],'y': torch.tensor([df_circle[1][i-4].iloc[0:300,-8+i].values])[0]}
            data[i]['test']= {'x': torch.tensor([df_circle[1][i-4].iloc[300:,0:-8].values])[0],'y': torch.tensor([df_circle[1][i-4].iloc[300:,-8+i].values])[0]}

    # Arrow
    for i, action in enumerate(actions[1]):
        if i < 4:
            data[i+8] = {}
            data[i+8]['name'] = action
            data[i+8]['n_outputs'] = 2
            data[i+8]['train']= {'x': torch.tensor([df_arrow[0][i].iloc[0:700,0:-8].values])[0],'y': torch.tensor([df_arrow[0][i].iloc[0:700,-8+i].values])[0]}
            data[i+8]['test']= {'x': torch.tensor([df_arrow[0][i].iloc[700:,0:-8].values])[0],'y': torch.tensor([df_arrow[0][i].iloc[700:,-8+i].values])[0]}
        else:
            data[i+8] = {}
            data[i+8]['name'] = action
            data[i+8]['n_outputs'] = 2
            data[i+8]['train']= {'x': torch.tensor([df_arrow[1][i-4].iloc[0:300,0:-8].values])[0],'y': torch.tensor([df_arrow[1][i-4].iloc[0:300,-8+i].values])[0]}
            data[i+8]['test']= {'x': torch.tensor([df_arrow[1][i-4].iloc[300:,0:-8].values])[0],'y': torch.tensor([df_arrow[1][i-4].iloc[300:,-8+i].values])[0]}

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
