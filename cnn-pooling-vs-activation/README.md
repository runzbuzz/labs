# Pooling and Activation Comparison in CNN

Comparing pooling, filters, activations functions on hard-written digits on a modified LeNet5

# Parameter Cube

pooling: max pooling vs avg pooling
activation functions:

- ReLu
- Tanh
- Sigmoid
- Swish
- GELU
- Leaky ReLu
- ELU

# Training

To train the network, you can either do it for a specific set of configurations
(e.g., Average Pooling and ReLu Activation) something like this:

```
python main.py --dataset mnist --pool avg --activation relu --seeds 1
```

or for the whole supported configurations (pooling and activation functions).

```
python main.py --dataset mnist --pool all --activation all --seeds 1 2 3
python main.py --dataset fashion_mnist --pool all --activation all --seeds 1 2 3
python main.py --dataset kmnist --pool all --activation all --seeds 1 2 3
python main.py --dataset emnist-balanced --pool all --activation all --seeds 1 2 3
```

## To aggregate the results

```
python main.py --dataset mnist  --agg --verbose 2 --overwrite n
python main.py --dataset fashion_mnist --agg --verbose 2 --overwrite n
python main.py --dataset kmnist --agg --verbose 2 --overwrite n
python main.py --dataset emnist-balanced --agg --verbose 2 --overwrite n
```

## To generate the charts

The charts data are designed to be used by charts.js.

```
python main.py --charts --dataset mnist
python main.py --charts --dataset fashion_mnist
python main.py --charts --dataset kmnist
python main.py --charts --dataset emnist-balanced
```

