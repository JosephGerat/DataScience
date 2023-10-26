import random

from neural_networks import sigmoid
from probability import inverse_normal_cdf

Tensor = list

from typing import List, Callable


def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []

    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]

    return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1, 2], [3, 4], [5, 6]]) == [3, 2]

def is_1d(tensor: Tensor) -> bool:
    return not isinstance(tensor[0], list)

assert is_1d([1, 2, 3])
assert not is_1d([[1], [2]])

def tensor_sum(tensor: Tensor) -> float:
    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i) for tensor_i in tensor)

assert tensor_sum([1, 2, 5]) == 8
assert tensor_sum([[1], [5], [20]]) == 26

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]

assert tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)

assert zeros_like([1, 2, 3]) == [0, 0, 0]
assert zeros_like([[2, 0], [4, 5]]) == [[0, 0], [0, 0]]

def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor
                   ) -> Tensor:
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i) for t1_i, t2_i in zip(t1, t2)]

import operator
assert tensor_combine(operator.add, [1, 2, 3], [3, 6, 11]) == [4, 8, 14]
assert tensor_combine(operator.mul, [1, 2, 3], [3, 6, 11]) == [3, 12, 33]

from typing import Iterable, Tuple

class Layer:

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        """ Returns the parameter of this layer"""
        return ()

    def grads(self) -> Iterable[Tensor]:
        """ Returns the gradients, in the same order as params(). """
        return ()


class Sigmoid(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad, self.sigmoids, gradient)

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]



def random_normal(*dims: int, mean: float = 0.0, variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random()) for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance) for _ in range(dims[0])]

assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean=10)) == [5, 6]

def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f'unknown  init: {init}')

from linear_algebra import dot

class Linear(Layer):

    def __init__(self, input_dim: int, output_dim: int, init: str = 'xavier') -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w = random_tensor(output_dim, input_dim, init=init)
        self.b = random_tensor(output_dim, init=init)

    def forward(self, input: Tensor) -> Tensor:
        self.input = input

        return [dot(input, self.w[o]) + self.b[o] for o in range(self.output_dim)]

    def backward(self, gradient: Tensor) -> Tensor:
        self.b_grad = gradient
        self.w_grad = [[self.input[i] * gradient[o] for i in range(self.input_dim)] for o in range(self.output_dim)]

        return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim)) for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]

class Sequential(Layer):

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> Iterable[Tensor]:
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        return (grad for layer in self.layers for grad in layer.grads())

xor_net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=1),
    Sigmoid()
])

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class SSE(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        squared_errors = tensor_combine(lambda predicted, actual: (predicted - actual) ** 2, predicted, actual)
        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(lambda predicted, actual: 2 * (predicted - actual), predicted, actual)

class Optimizer:
    def step(self, layer: Layer) -> None:
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            param[:] = tensor_combine(lambda param, grad: param - grad * self.lr, param, grad)

class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []

    def step(self, layer: Layer) -> None:
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates, layer.params(), layer.grads()):
            # apply momentum
            update[:] = tensor_combine(lambda u, g: self.mo * u + (1 - self.mo) * g, update, grad)

            # then take a gradient step
            param[:] = tensor_combine(lambda p, u: p - self.lr * u, param, update)

xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

random.seed(0)
net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=1)
])

optimizer = GradientDescent(learning_rate=0.1)
loss = SSE()

for epoch in range(3000):
    epoch_loss = 0.0
    for x, y in zip(xs, ys):
        predicted = net.forward(x)
        epoch_loss += loss.loss(predicted, y)
        gradient = loss.gradient(predicted, y)
        net.backward(gradient)

        optimizer.step(net)
    print(f'Loss {epoch_loss:.3f}')