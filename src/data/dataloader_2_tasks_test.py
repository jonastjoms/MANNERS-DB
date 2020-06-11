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

    data= {}
    task_outputs = []
    size = [1,29]

    data[0]={}
    data[0]['name']='Circle'
    data[0]['n_outputs'] = 16
    data[1]={}
    data[1]['name'] = 'Arrow'
    data[1]['n_outputs'] = 16

    data[0]['test']={'x': [],'y': []}
    data[1]['test']={'x': [],'y': []}

    # Circle:
    count = 0
    for i, row in df_circle.iterrows():
        x = torch.tensor(row.values[0:-8])
        y = torch.tensor(row.values[-8:])
        # Test data
        data[0]['test']['x'].append(x)
        data[0]['test']['y'].append(y)

    # Arrow:
    count = 0
    for i, row in df_arrow.iterrows():
        x = torch.tensor(row.values[0:-8])
        y = torch.tensor(row.values[-8:])
        # Test data
        data[1]['test']['x'].append(x)
        data[1]['test']['y'].append(y)


    # "Unify" and save
    for n in [0,1]:
        for s in ['test']:
            data[n][s]['x'] = torch.stack(data[n][s]['x'])
            data[n][s]['y'] = torch.stack(data[n][s]['y'])

    # Others
    n=0
    for t in data.keys():
        task_outputs.append((t, data[t]['n_outputs']))
        n += data[t]['n_outputs']
    data['ncla'] = n

    return data

########################################################################################################################
