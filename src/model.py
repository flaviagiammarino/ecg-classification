import warnings
import random
import time
import torch
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')

from src.modules import LSTM_FCN, FCN

class Model():
    
    def __init__(self, features, units, dropout, filters, kernel_sizes, num_classes, model_type='baseline', seed=42):
        '''
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
        
        model_type: str.
            Model type, either 'baseline' or 'proposed' (default = 'baseline').
        
        seed: int.
            Random seed.
        '''
        
        # save the inputs
        self.seed = seed
        self.model_type = model_type
        
        # check if GPU is available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.use_data_parallel = torch.cuda.device_count() > 1
        self.kwargs = {}
        if self.use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
        
        # build and save the model
        self.set_seed()
        if model_type == 'baseline':
            self.model = FCN(features, filters, kernel_sizes, num_classes)
        elif model_type == 'proposed':
            self.model = LSTM_FCN(features, units, dropout, filters, kernel_sizes, num_classes)
        self.model.to(self.device)
        if self.use_data_parallel:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        print(f'number of parameters: {format(sum(p.numel() for p in self.model.parameters()), ",.0f")}')
    
    def set_seed(self):
        '''
        Fix all random seeds, for reproducibility.
        '''
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.seed)

    def fit(self, x_train, y_train, x_test, y_test, learning_rate, batch_size, epochs, verbose=True):
        '''
        Parameters:
        __________________________________
        x_train: np.ndarray.
            Training time series.
        
        y_train: np.ndarray.
            Training labels.
        
        x_test: np.ndarray.
            Test time series.
        
        y_test: np.ndarray.
            Test labels.
            
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''
        
        # create the training dataloader
        self.set_seed()
        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).float()
            ),
            batch_size=batch_size,
            shuffle=True,
            **self.kwargs
        )
        
        # create the test dataloader
        test_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.from_numpy(x_test).float(),
                torch.from_numpy(y_test).float()
            ),
            batch_size=batch_size,
            shuffle=False,
            **self.kwargs
        )
        
        # instantiate the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # define the loss function
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # train the model
        if self.use_data_parallel:
            print(f'training on CUDA {", ".join([str(d) for d in self.model.device_ids])}')
        else:
            print(f'training on {str(self.device).upper()}')
        self.history = []
        for epoch in range(epochs):
            
            # run the training step
            train_loss = 0
            train_start = time.time()
            self.model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self.model(data.to(self.device))
                loss = loss_fn(output, target.to(self.device))
                train_loss += loss
                loss.backward()
                optimizer.step()
            train_end = time.time()
            train_loss /= len(train_loader.dataset)
            
            # run the validation step
            test_loss = 0
            test_start = time.time()
            self.model.eval()
            for data, target in test_loader:
                with torch.no_grad():
                    output = self.model(data.to(self.device))
                test_loss += loss_fn(output, target.to(self.device))
            test_end = time.time()
            test_loss /= len(test_loader.dataset)
            
            # save the results
            self.history.append({
                'epoch': 1 + epoch,
                'train_time': format(train_end - train_start, '.2f'),
                'train_loss': format(train_loss, '.6f'),
                'test_time': format(test_end - test_start, '.2f'),
                'test_loss': format(test_loss, '.6f'),
            })
            
            # display the results
            if verbose:
                print(", ".join([f'{k}: {v}' for k, v in self.history[-1].items()]))
        
        self.history = pd.DataFrame(data=self.history).astype(float)
        print(f'total training time: {self.history["train_time"].sum() // 60} minutes')
    
    def predict(self, x, batch_size=512):
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (batch size, features, timesteps).
        '''
        
        # create the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.from_numpy(x).float()
            ),
            batch_size=batch_size,
            shuffle=False,
            **self.kwargs
        )
        
        # get the predicted probabilities
        self.model.eval()
        probs = []
        for data in dataloader:
            with torch.no_grad():
                probs.append(torch.nn.functional.sigmoid(self.model(data[0].to(self.device))))
        probs = torch.cat(probs).detach().cpu().numpy()
        
        return probs
