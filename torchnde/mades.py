import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------------------
class LinearMasked(nn.Module):
    def __init__(self, in_features, out_features, num_input_features, bias=True):
        """

        Parameters
        ----------
        in_features : int
        out_features : int
        num_input_features : int
            Number of features of the models input X.
            These are needed for all masked layers.
        bias : bool
        """
        super(LinearMasked, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.num_input_features = num_input_features

        assert (
            out_features >= num_input_features
        ), "To ensure autoregression, the output there should be enough hidden nodes. h >= in."

        # Make sure that d-values are assigned to m
        # d = 1, 2, ... D-1
        d = set(range(1, num_input_features))
        c = 0
        while True:
            c += 1
            if c > 10:
                break
            # m function of the paper. Every hidden node, gets a number between 1 and D-1
            #if num_input_features > 1:
            self.m = torch.randint(min(1, num_input_features-1), num_input_features, size=(out_features,)).type(
                torch.int32
            )
            #else:
            #    self.m = torch.ones(size=(out_features,))
            if len(d - set(self.m.numpy())) == 0:
                break

        self.register_buffer(
            "mask", torch.ones_like(self.linear.weight).type(torch.uint8)
        )

    def set_mask(self, m_previous_layer):
        """
        Sets mask matrix of the current layer.

        Parameters
        ----------
        m_previous_layer : tensor
            m values for previous layer layer.
            The first layers should be incremental except for the last value,
            as the model does not make a prediction P(x_D+1 | x_<D + 1).
            The last prediction is P(x_D| x_<D)
        """
        self.mask[...] = (m_previous_layer[:, None] <= self.m[None, :]).T

    def forward(self, x):
        if self.linear.bias is None:
            b = 0
        else:
            b = self.linear.bias

        return F.linear(x, self.linear.weight * self.mask, b)

def set_mask_output_layer(layer, m_previous_layer):
    # Output layer has different m-values.
    # The connection is shifted one value to the right.
    layer.m = torch.arange(0, layer.num_input_features)
    layer.set_mask(m_previous_layer)
    return layer


def set_mask_input_layer(layer):
    m_input_layer = torch.arange(1, layer.num_input_features + 1)
    m_input_layer[-1] = 1e9
    layer.set_mask(m_input_layer)
    return layer

class ConditionalGaussianMADE(nn.Module):
    # Don't use ReLU, so that neurons don't get nullified.
    # This makes sure that the autoregressive test can verified
    def __init__(self, in_features, out_features, hidden_layers=[32, 32], activation = 'elu'):

        #if out_features < 2:
        #    raise ValueError('The class currently only supports at least 2 out features')

        super().__init__()
        if activation.lower() == 'elu':
            self.act = F.elu
        elif activation.lower() == 'tanh':
            self.act = F.tanh
        elif activation.lower() == 'sigmoid':
            self.act = F.sigmoid
        else:
            raise ValueError("Only sigmoid, elu or tanh are implemented as activation functions")

        self.in_features = in_features
        self.out_features = out_features

        self.lx = nn.Linear(in_features, hidden_layers[0], bias = False)
        firstlayer = LinearMasked(out_features, hidden_layers[0], out_features)
        firstlayer = set_mask_input_layer(firstlayer)
        m_previous_layer = firstlayer.m
        self.layers = [firstlayer]
        for l0, l1 in zip(hidden_layers[:-1], hidden_layers[1:]):
            newlayer = LinearMasked(l0, l1, out_features)
            newlayer.set_mask(m_previous_layer)
            m_previous_layer = newlayer.m
            self.layers.append(newlayer)
        self.layer_mean = LinearMasked(hidden_layers[-1], out_features, out_features)
        self.layer_mean = set_mask_output_layer(self.layer_mean, m_previous_layer)
        self.layer_sigma = LinearMasked(hidden_layers[-1], out_features, out_features)
        self.layer_sigma = set_mask_output_layer(self.layer_sigma, m_previous_layer)
        #self.layers.set_mask_last_layer()

    def forward(self, x, y):
        out = torch.add(self.lx(x), self.layers[0](y))
        out = self.act(out)
        for layer in self.layers[1:]:
            out = layer.forward(out)
            out = self.act(out)

        mean = self.layer_mean(out)
        sigma = self.layer_sigma(out)
        return mean, sigma

    def eval(self, x, y):
        mean, sigma = self.forward(x, y)
        u = torch.exp(0.5 * sigma) * (y - mean)

        # log likelihoods
        return -0.5 * (self.out_features * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(u ** 2 - sigma, axis=1, keepdim=True))

    def gen(self, x, n_samples=1, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        y = torch.zeros([n_samples, self.out_features])
        u = torch.randn(n_samples, self.out_features) if u is None else u

        for i in range(1, self.in_features + 1):
            mean, sigma = self.forward(x, y)
            #idx = np.argwhere(self.input_order == i)[0, 0]]
            y = mean + torch.exp(torch.minimum(-0.5 * sigma, torch.tensor(10.0))) * u

        return y

    def train(self, x, y, n_epochs = 10000, lr = 1e-3, batch_size = 32, verbose = True, print_every=100,  optimizer = None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        dataset = TensorDataset(x, y)
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            if verbose and epoch % print_every == 0 and epoch > 0: print(epoch, loss)
            for x, y in trainloader:
                loss = -self.eval(x, y).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

if __name__ == "__main__":
    input_size = 1
    output_size = 1
    m = ConditionalGaussianMADE(in_features=input_size, out_features=output_size, hidden_layers=[32, 32])
    xx = torch.rand((1000, input_size))
    d = torch.randn((1000, output_size))
    yy = torch.sum(xx, axis=1, keepdim=True)*(1+0.1*d)
    print(xx.shape, yy.shape)
    xx.requires_grad = True

    m.train(xx,yy,n_epochs=100)

    print(m.gen(xx[0]))
    print(yy[0])