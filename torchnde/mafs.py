import torch
import torch.nn as nn
import torch.nn.functional as F
from mades import ConditionalGaussianMADE
from torch.utils.data import DataLoader, TensorDataset

# TODO: Custom number of layer
# TODO: Uniform notation (n_inputs or n_features, etc)
# TODO: Add batch_norm
# TODO: Trainer for other MADES
# TODO: Make MAF

class ConditionalMaskedAutoregressiveFlow(nn.Module):
    """
    Implements a Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_inputs, n_outputs, n_mades, hidden_layers = [32,32], activation = 'elu', batch_norm=True):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param n_mades: number of mades in the flow
        :param batch_norm: whether to use batch normalization between mades in the flow
        :param output_order: order of outputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        :param output: theano variable to serve as output; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_mades = n_mades
        #self.batch_norm = batch_norm

        self.mades = []
        self.parameters = []

        for i in range(n_mades):
            # create a new made
            made = ConditionalGaussianMADE(n_inputs, n_outputs, hidden_layers, activation=activation)
            self.mades.append(made)
            self.parameters = self.parameters + list(made.parameters())

    def eval(self, x, y):
        logdet_dudy = 0.0
        u = y
        for made in self.mades:
            mean, sigma = made.forward(x, u)
            #print(u.shape, mean.shape)
            u = torch.exp(0.5 * sigma) * (u - mean)
            logdet_dudy += 0.5 * torch.sum(sigma, axis=1, keepdim=True)

        return -0.5*self.n_outputs * torch.log(torch.tensor(2*torch.pi)) - 0.5*torch.sum(u**2, axis = 1, keepdim=True) + logdet_dudy

    def gen(self, x, n_samples=1, u=None):
        y = torch.randn(n_samples, self.n_outputs) if u is None else u

        #if getattr(self, 'batch_norm', False):

        for made in self.mades[::-1]:
            y = made.gen(x, n_samples, y)

        return y

    def train(self, x, y, n_epochs=10000, lr=1e-4, batch_size=32, print_every = 100, verbose = True, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters, lr=lr)

        dataset = TensorDataset(x, y)
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            if verbose and epoch%print_every == 0 and epoch>0: print(epoch, loss)
            for x, y in trainloader:
                loss = -self.eval(x, y).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, 1.0)
                optimizer.step()

if __name__ == "__main__":
    input_size = 10
    output_size = 3
    m = ConditionalMaskedAutoregressiveFlow(input_size, output_size, hidden_layers = [32, 32], n_mades = 2)
    xx = torch.ones((1, input_size))
    yy = torch.ones((1, output_size))
    xx.requires_grad = True

    m.train(xx,yy,n_epochs=100)
    print(m.gen(xx))
    print(yy)