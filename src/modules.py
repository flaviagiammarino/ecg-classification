import torch
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')

class RecurrentBranch(torch.nn.Module):
    '''
    Recurrent branch.
    
    Parameters:
    __________________________________
    features: int.
      The dimension of each (multivariate) time series.
    
    units: list of int.
      The length of the list corresponds to the number of recurrent blocks, the
      items in the list are the number of units of the LSTM layer in each block.
    
    dropout: float.
      Dropout rate to be applied after each recurrent block.
    '''
    def __init__(self, features, units, dropout):
        super(RecurrentBranch, self).__init__()
        
        # check the inputs
        if type(units) != list:
            raise ValueError(f'The number of units should be provided as a list.')
        
        # build the model
        modules = OrderedDict()
        for i in range(len(units)):
            modules[f'LSTM_{i}'] = torch.nn.LSTM(
                input_size=features if i == 0 else units[i - 1],
                hidden_size=units[i],
                batch_first=True
            )
            modules[f'Lambda_{i}'] = Lambda(f=lambda x: x[0])
            modules[f'ReLU_{i}'] = torch.nn.ReLU()
            modules[f'Dropout_{i}'] = torch.nn.Dropout(p=dropout)
        self.model = torch.nn.Sequential(modules)
    
    def forward(self, x):
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (batch size, features, timesteps).
        '''
        return self.model(torch.transpose(x, 2, 1))[:, -1, :]


class ConvolutionalBranch(torch.nn.Module):
    '''
    Convolutional branch.

    Parameters:
    __________________________________
    features: int.
        The dimension of each (multivariate) time series.

    filters: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the number of filters (or channels) of the convolutional layer in each block.

    kernel_sizes: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the kernel size of the convolutional layer in each block.
    '''
    def __init__(self, features, filters, kernel_sizes):
        super(ConvolutionalBranch, self).__init__()
        
        # check the inputs
        if len(filters) == len(kernel_sizes):
            blocks = len(filters)
        else:
            raise ValueError(f'The number of filters and kernel sizes must be the same.')
        
        # build the model
        modules = OrderedDict()
        for i in range(blocks):
            modules[f'Conv1d_{i}'] = torch.nn.Conv1d(
                in_channels=features if i == 0 else filters[i - 1],
                out_channels=filters[i],
                kernel_size=(kernel_sizes[i],),
                padding='same',
                bias=False
            )
            modules[f'BatchNorm1d_{i}'] = torch.nn.BatchNorm1d(
                num_features=filters[i],
                eps=0.001,
                momentum=0.99
            )
            modules[f'ReLU_{i}'] = torch.nn.ReLU()
        self.model = torch.nn.Sequential(modules)
    
    def forward(self, x):
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (batch size, features, timesteps).
        '''
        return self.model(x)


class FCN(torch.nn.Module):
    '''
    FCN model of Wang et al. 2017 (10.1109/IJCNN.2017.7966039).
    
    Parameters:
    __________________________________
    features: int.
        The dimension of each (multivariate) time series.

    filters: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the number of filters (or channels) of the convolutional layer in each block.

    kernel_sizes: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the kernel size of the convolutional layer in each block.

    num_classes: int.
        Number of classes.
    '''
    def __init__(self, features, filters, kernel_sizes, num_classes):
        super(FCN, self).__init__()
        
        # convolutional branch
        self.fcn = ConvolutionalBranch(features=features, filters=filters, kernel_sizes=kernel_sizes)
        
        # classification head of Strodthoff et al. 2021 (10.1109/JBHI.2020.3022989)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=2 * filters[-1])
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=128)
        
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.50)
        
        self.linear1 = torch.nn.Linear(in_features=2 * filters[-1], out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=num_classes)
    
    def forward(self, x):
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (batch size, features, timesteps).
        
        Returns:
        __________________________________
        y: torch.Tensor.
            Logits, tensor with shape (batch size, num_classes).
        '''
        h = self.fcn(x)
        h = torch.squeeze(torch.concat([self.avg_pool(h), self.max_pool(h)], dim=1))
        h = self.linear1(self.dropout1(self.batch_norm1(h)))
        h = torch.nn.functional.relu(h)
        h = self.linear2(self.dropout2(self.batch_norm2(h)))
        return h


class LSTM_FCN(torch.nn.Module):
    '''
    LSTM-FCN model of Karim et al. 2018 (10.1109/ACCESS.2017.2779939).
    
    Parameters:
    __________________________________
    features: int.
        The dimension of each (multivariate) time series.

    units: list of int.
        The length of the list corresponds to the number of recurrent blocks, the items in the
        list are the number of units of the LSTM layer in each block.

    dropout: float.
        Dropout rate to be applied after each recurrent block.

    filters: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the number of filters (or channels) of the convolutional layer in each block.

    kernel_sizes: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the kernel sizes of the convolutional layer in each block.

    num_classes: int.
        Number of classes.
    '''
    def __init__(self, features, units, dropout, filters, kernel_sizes, num_classes):
        super(LSTM_FCN, self).__init__()
        
        # convolutional branch
        self.fcn = ConvolutionalBranch(features=features, filters=filters, kernel_sizes=kernel_sizes)
        
        # recurrent branch
        self.lstm = RecurrentBranch(features=features, units=units, dropout=dropout)
        
        # convolutional branch
        self.fcn = ConvolutionalBranch(features=features, filters=filters, kernel_sizes=kernel_sizes)
        
        # classification head of Strodthoff et al. 2021 (10.1109/JBHI.2020.3022989)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=2 * filters[-1] + units[-1])
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=128)
        
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.50)
        
        self.linear1 = torch.nn.Linear(in_features=2 * filters[-1] + units[-1], out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=num_classes)


    def forward(self, x):
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (batch size, features, timesteps).

        Returns:
        __________________________________
        y: torch.Tensor.
            Logits, tensor with shape (batch size, num_classes).
        '''
        h = self.fcn(x)
        h = torch.squeeze(torch.concat([self.avg_pool(h), self.max_pool(h)], dim=1))
        h = torch.concat([h, self.lstm(x)], dim=-1)
        h = self.linear1(self.dropout1(self.batch_norm1(h)))
        h = torch.nn.functional.relu(h)
        h = self.linear2(self.dropout2(self.batch_norm2(h)))
        return h


class Lambda(torch.nn.Module):
    '''
    Lambda layer.
    '''
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)