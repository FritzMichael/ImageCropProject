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
    
    def forward(self, x, crop_dict):

        y = torch.zeros((x.size()[0], 4, x.size()[2], x.size()[3]), device='cuda')

        for sample, x_imag, current_cd in zip(y,x,crop_dict):
            sample[0] = x_imag
            sample[1][current_cd['top']:current_cd['bottom']+1,current_cd['left']:current_cd['right']+1] = 1
            sample[2] = torch.linspace(start=-1, end=1, steps=x.size()[3], device='cuda').repeat(x.size()[2],1)
            sample[3] = torch.transpose(torch.linspace(start=-1, end=1, steps=x.size()[2], device='cuda').repeat(x.size()[3],1),0,1)

        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(y)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        
        #target_mask = torch.squeeze(x)[1].to(dtype=torch.bool)
        #pred = torch.squeeze(pred)[target_mask]

        return pred, y[:,1,:,:]