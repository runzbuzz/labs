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

# Calling it

Example:

```
python main.py --dataset mnist --pool avg --activation relu --seeds 1
```

```
python main.py --dataset mnist --pool all --activation all --seeds 1 2 3
python main.py --dataset fashion_mnist --pool all --activation all --seeds 1 2 3
python main.py --dataset kmnist --pool all --activation all --seeds 1 2 3
python main.py --dataset emnist-balanced --pool all --activation all --seeds 1 2 3
```

## To aggregate the results

```
python main.py --dataset mnist --pool all --activation all --agg --verbose 2 --overwrite a
python main.py --dataset fashion_mnist --pool all --activation all --agg --verbose 2 --overwrite a
python main.py --dataset kmnist --pool all --activation all --agg --verbose 2 --overwrite a
python main.py --dataset emnist-balanced --pool all --activation all --agg --verbose 2 --overwrite a
```

## To generate the charts

The charts file is to be used by charts.js
Note: this command does not filter based on batch_size nor on epochs.

```
python main.py --charts --dataset mnist
python main.py --charts --dataset fashion_mnist
python main.py --charts --dataset kmnist
python main.py --charts --dataset emnist-balanced
```