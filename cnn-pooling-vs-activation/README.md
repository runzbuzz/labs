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

# Calling the different settings

```
python main.py --pool avg --activation relu ;
python main.py --pool max --activation relu ;

python main.py --pool avg --activation tanh ;
python main.py --pool max --activation tanh ;

python main.py --pool avg --activation sigmoid ;
python main.py --pool max --activation sigmoid 

python main.py --pool avg --activation swish ;
python main.py --pool max --activation swish ;

python main.py --pool avg --activation gelu ;
python main.py --pool max --activation gelu ;

python main.py --pool avg --activation leaky_relu ;
python main.py --pool max --activation leaky_relu ;

python main.py --pool avg --activation elu ;
python main.py --pool max --activation elu ;


```

## To generate the charts

The charts file is to be used by charts.js

```
python main.py --charts
```