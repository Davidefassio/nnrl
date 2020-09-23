# nn (Neural network)
Neural network implementation in JS.

## Methods overall
### Neural network
```constructor``` Initialize the model and set the learning rate to its default value. No inputs or outputs.

```add_layer``` Add a neuron layer to the model. Inputs: neurons = number of neurons in that layer (must be > 0), activ_func = a string that contain the name of the activation function used to activate that layer(must be one of these: "none", "relu", "softplus", "sigmoid", "tanh", "softmax"). 

```set_learning_rate``` Change the learning rate of the model if the number given is > 0. Input: a real number. Output: true if the input is positive and the learning rate is being changed, false otherwise.

```guess``` Return a guess based on the input vector given. Input: 1D vector. Output: 1D vector.

```backprop``` TODO

### Activation functions
```activate``` Apply the activation function to the layer. Input: x = the vector containing the layer to modify, i = the number of the layer (0 = input layer, ...).

```relu``` ReLU (Rectified Linear Unit). output = max(0, input). Input: 1D vector.

```softplus``` Smooth version of ReLU. output = ln(1 + exp(input)). Input: 1D vector.

```sigmoid```

```tanh```

```softmax```

```der_relu```

```der_softplus```

```der_sigmoid```

```der_tanh```

```der_softmax```

### Random generators
```rand```

```norm_distr```

### Linear algebra
```vec_sum```

```mat_sum```

```vec_sumTo```

```mat_sumTo```

```vec_sub```

```mat_sub```

```vec_subTo```

```mat_subTo```

```vec_kmult```

```mat_kmult```

```dotmm```

```dotvm```

```dotvv```

```transpose```
