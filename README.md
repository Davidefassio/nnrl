# nn (Neural network)
Neural network implementation in JS.

## Methods overall
### Neural network
```constructor```

```add_layer``` Add a neuron layer to the model. The parameters are: neurons = number of neurons in that layer (must be > 0), activ_func = a string that contain the name of the activation function used to activate that layer(must be one of these: "none", "relu", "softplus", "sigmoid", "tanh", "softmax"). 

```set_learning_rate```

```guess``` Return a guess based on the input given. The output and the input must be 1D vector.

```backprop```

### Activation functions
```activate```

```relu```

```softplus```

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
