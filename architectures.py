import torch

class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(SimpleCNN, self).__init__()
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size/2)))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size/2))
        self.final_layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=True, padding=int(3/2) )
    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        pred = self.final_layer(pred)
        
        #target_mask = torch.squeeze(x)[1].to(dtype=torch.bool)
        #pred = torch.squeeze(pred)[target_mask]

        return pred

class ComplexCNN(torch.nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

        self.coarseCnn = SimpleCNN(n_in_channels=4, kernel_size=15)
        self.fineCnn = SimpleCNN(n_in_channels=4, kernel_size=3)

        self.outputCnn = SimpleCNN(n_in_channels=6)

    def forward(self, x):
        coarseOutput = self.coarseCnn(x)
        fineOutput = self.fineCnn(x)

        x = torch.cat((torch.unsqueeze(x[:,0,:,:],1),torch.unsqueeze(x[:,1,:,:],1),torch.unsqueeze(x[:,2,:,:],1),torch.unsqueeze(x[:,3,:,:],1), coarseOutput, fineOutput))
        x = x.permute(1,0,2,3)

        return self.outputCnn(x)
