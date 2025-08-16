

import os
import torch
import numpy as np
from torch.utils.data  import  TensorDataset,DataLoader



def read_data_normalization(path_to_jena_Dataset, train_samples_percent= 50,val_samples_percent = 25):

    path_data = os.path.join(path_to_jena_Dataset)  
    with open(path_data) as jena_file:
        data = jena_file.read()


    lines = data.split("\n")
    header_line = lines[0].split(',')
    data_lines = lines[1:]

    temperature = np.zeros(len(data_lines))
    raw_data = np.zeros((len(data_lines), len(header_line) - 1))

    for i , line in enumerate(data_lines):

        values = [float(element)   for element in  line.split(',')[1:]]
        temperature[i] = values[1]
        raw_data[i ,:] = values[:]


    number_train_samples = int((train_samples_percent/100) * len(raw_data))  
    number_val_samples =  int((val_samples_percent/100) * len(raw_data)) 
    number_test_samples = len(raw_data) - number_train_samples - number_val_samples


    mean = raw_data[:number_train_samples].mean(axis=0)
    raw_data -= mean

    std = raw_data[:number_train_samples].std(axis=0)
    raw_data /= std


    return (raw_data, temperature, number_train_samples, number_val_samples, number_test_samples)





def make_sequencer_and_dataloader(X, y, seq_length = 120, pred_length = 24, sampling_rate = 6, batch_size = 512, shuffle = False):

    delay = sampling_rate * (seq_length + pred_length - 1)
    #convert to tensor object
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32)

    X_seq = torch.stack([ X[i:i+seq_length ,:]      for i in  range(len(X) - delay ) ])
    y_seq = torch.stack([y[i+seq_length]            for i in  range(len(X) - delay )  ])
    

    #create tensor_dataset
    tensor_data = TensorDataset(X_seq, y_seq)

    return  DataLoader(tensor_data ,batch_size, shuffle=shuffle )











