import torch
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')


class RecurrentBranch(torch.nn.Module):
    '''
    Recurrent branch.
    
    Parameters:
    __________________________________
    timesteps: int.
        The length of each time series.
        
    features: int.
        The dimension of each time series.
    
    hidden_size: int.
        The number of units of each LSTM layer.
    
    num_layers: int.
        The number of LSTM layers.
    
    dropout: float.
        The dropout rate applied after each LSTM layer.
    '''
    def __init__(self, timesteps, hidden_size, num_layers, dropout):
        super(RecurrentBranch, self).__init__()
        
        # build the model
        modules = OrderedDict()
        for i in range(num_layers):
            modules[f'LSTM_{i}'] = torch.nn.LSTM(
                input_size=timesteps if i == 0 else hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )
            modules[f'Lambda_{i}'] = Lambda(f=lambda x: x[0])
            modules[f'Dropout_{i}'] = torch.nn.Dropout(p=dropout)
        self.model = torch.nn.Sequential(modules)
    
    def forward(self, x):
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (batch size, features, timesteps).
            Note the dimension shuffling.
        '''
        return self.model(x)[:, -1, :]


class ConvolutionalBranch(torch.nn.Module):
    '''
    Convolutional branch.

    Parameters:
    __________________________________
    features: int.
        The dimension of each time series.

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
        The dimension of each time series.

    filters: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the number of filters (or channels) of the convolutional layer in each block.

    kernel_sizes: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the kernel size of the convolutional layer in each block.

    units: int.
        The number of units of the classification head.
    
    num_classes: int.
        The number of classes.
    '''
    def __init__(self, features, filters, kernel_sizes, units, num_classes):
        super(FCN, self).__init__()
        
        # convolutional branch
        self.fcn = ConvolutionalBranch(
            features=features,
            filters=filters,
            kernel_sizes=kernel_sizes
        )
        
        # classification head of Strodthoff et al. 2021 (10.1109/JBHI.2020.3022989)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=2 * filters[-1])
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=units)
        
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.50)
        
        self.linear1 = torch.nn.Linear(in_features=2 * filters[-1], out_features=units)
        self.linear2 = torch.nn.Linear(in_features=units, out_features=num_classes)
    
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
    timesteps: int.
        The length of each time series.
        
    features: int.
        The dimension of each time series.
    
    hidden_size: int.
        The number of units of each LSTM layer.
    
    num_layers: int.
        The number of LSTM layers.
    
    dropout: float.
        The dropout rate applied after each LSTM layer.
        
    filters: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the number of filters (or channels) of the convolutional layer in each block.

    kernel_sizes: list of int.
        The length of the list corresponds to the number of convolutional blocks, the items in the
        list are the kernel sizes of the convolutional layer in each block.

    units: int.
        The number of units of the classification head.
        
    num_classes: int.
        The number of classes.
    '''
    def __init__(self, timesteps, features, hidden_size, num_layers, dropout, filters, kernel_sizes, units, num_classes):
        super(LSTM_FCN, self).__init__()
        
        # convolutional branch
        self.fcn = ConvolutionalBranch(
            features=features,
            filters=filters,
            kernel_sizes=kernel_sizes
        )
        
        # recurrent branch
        self.lstm = RecurrentBranch(
            timesteps=timesteps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # classification head of Strodthoff et al. 2021 (10.1109/JBHI.2020.3022989)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=2 * filters[-1])
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=units + hidden_size)
        
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.50)
        
        self.linear1 = torch.nn.Linear(in_features=2 * filters[-1], out_features=units)
        self.linear2 = torch.nn.Linear(in_features=units + hidden_size, out_features=num_classes)

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
        h = torch.concat([h, self.lstm(x)], dim=-1)
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