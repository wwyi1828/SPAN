import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, feature_dim, eps=1e-8, elementwise_affine=True):
        super(RMSNorm, self).__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.g = nn.Parameter(torch.ones(feature_dim))
        else:
            self.register_parameter('g', None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(torch.mean((x - mean) ** 2, dim=-1, keepdim=True) + self.eps)
        norm_x = (x - mean) / std

        if self.elementwise_affine:
            norm_x = self.g * norm_x

        return norm_x

class DynamicTanhNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, alpha_init_value=0.5):

        super().__init__()

        if isinstance(normalized_shape, (list, tuple)):
            self.normalized_shape = normalized_shape
            self.channels_last = True
        else:
            self.normalized_shape = (normalized_shape,)
            self.channels_last = True

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.channels_last:
            dims = tuple(range(-len(self.normalized_shape), 0))
        else:
            dims = tuple(range(1, len(self.normalized_shape) + 1))

        x = torch.tanh(self.alpha * x)

        if self.elementwise_affine:
            if self.channels_last:
                return x * self.weight + self.bias
            else:
                shape = [1, *self.normalized_shape] + [1] * (x.dim() - len(self.normalized_shape) - 1)
                return x * self.weight.view(shape) + self.bias.view(shape)
        else:
            return x

def create_activation(name: str, **kwargs):
    activation_class = getattr(nn, name)
    return activation_class(**kwargs)
