# nnrl (Neural network with reinforcement learning)
![alt text](logo.png)

Neural network and reinforcement learning implementation in JS.\
Still in development

## Methods overall
### Neural network
#### This set of methods is what you will use to create and use the neural network.
```constructor()```\
Initialize the model and set the learning rate to its default value.\
No inputs or outputs.

```add_layer(neurons, activ_func)```\
Add a neuron layer to the model.\
Input: neurons = number of neurons in that layer (must be > 0), activ_func = a string that contain the name of the activation function used to activate that layer(must be one of these: "none", "relu", "softplus", "sigmoid", "tanh", "softmax"). 

```set_learning_rate(l)```\
Change the learning rate of the model if the number given is > 0.\
Input: a real number. Output: true if the input is positive and the learning rate is being changed, false otherwise.

```set_loss_function(f)```

```guess(input)```\
Return a guess based on the input vector given using forward propagation.\
Input: 1D vector.\
Output: 1D vector.

```backprop``` TODO

### Loss functions (cost functions)
```mse(y_err, y_cor)```

```log_loss(y_err, y_cor)```

### Activation functions
#### A bunch of activation functions (funny stuff here), their derivatives and the functions to apply them.
```activate(x, i)```\
Apply the activation function to the layer.\
Input: x = the vector containing the layer to modify, i = the number of the layer (0 = input layer, ...).

```linear(x)```

```relu(x)```\
ReLU (Rectified Linear Unit). output = max(0, x).\
Input: 1D vector.

```softplus(x)```\
Smooth version of ReLU. output = ln(1 + exp(x)).\
Input: 1D vector.

```sigmoid(x)```\
Sigmoid function used to scale values in the range = (0, 1). output = 1 / (1 + exp(-x)).\
Input: 1D vector.

```tanh(x)```\
Hyperbolic tangent function used to scale values in the range = (-1, 1).\
output = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).\
Input: 1D vector.

```softmax(x)```\
Normalized exponential function also known as softargmax,\
used to normalize the values to a probability distribution with exponential proportional weight.\
Input: 1D vector.

#### All the derivatives of the activation functions take as input a vector ALREADY ACTIVATED with the same activation function.
```der_linear(x)

```der_relu(x)```\
Derivative of ReLU. output = 0 if x <= 0, 1 otherwise.\
Input: 1D vector. Output: 1D vector.

```der_softplus(x)```\
Derivative of softplus. output = 1 - 1/exp(x). \
Input: 1D vector. Output: 1D vector.

```der_sigmoid(x)```\
Derivative of sigmoid. output = x*(1 - x).\
Input: 1D vector. Output: 1D vector.

```der_tanh(x)```\
Derivative of hyperbolic tangent. output = 1 - x^2.\
Input: 1D vector. Output: 1D vector.

```der_softmax(x)```\
Derivative of softmax. For more detailed info search online.\
Input: 1D vector. Output: 2D matrix.

### Random generators
#### Methods used to set up the weights of the model to random values generated accordingly to "Xavier initialization" and "He initialization".
```rand(row, col, mean, variance)```\
Generate a 2D matrix with random values picked from a normal distribution.\
Input: row = number of rows of the matrix, col = number of columns of the matrix,\
mean = mean of the distribution, variance = variance of the distribution (variance = stdev ^ 2).\
Output: 2D matrix.

```norm_distr(mean, stdev)```\
Generate a number from a normal distribution.\
Input: mean = mean of the distribution, stdev = standard deviation of the distribution (stdev = sqrt(variance)).\
Output: a real value number.

### Linear algebra
#### A bunch of vector and matrix operation used in the neural network (nothing funny here XD).
```v_sumAll(v)```

```m_sumAll(m)```

```v_sum(v1, v2)```\
Sum two vectors without modifying them and return a vector containing the result.\
Input: two 1D vectors.\
Output: 1D vector.

```m_sum(m1, m2)```\
Sum two matrices without modifying them and return a matrix containing the result.\
Input: two 2D matrices.\
Output: 2D matrix.

```v_sumTo(v, v2)```\
Sum the second vector to the first one, the first vector is therefore modified.\
Input: two 1D vectors.

```m_sumTo(m, m2)```\
Sum the second matrix to the first one, the first matrix is therefore modified.\
Input: two 2D matrices.

```v_sub(v1, v2)```\
Subtract two vectors without modifying them and return a vector containing the result.\
Input: two 1D vectors.\
Output: 1D vector.

```m_sub(m1, m2)```\
Subtract two matrices without modifying them and return a matrix containing the result.\
Input: two 2D matrices.\
Output: 2D matrix.

```v_subTo(v, v2)```\
Subtarct the second vector to the first one, the first vector is therefore modified.\
Input: two 1D vectors.

```m_subTo(m, m2)```\
Subtract the second matrix to the first one, the first matrix is therefore modified.\
Input: two 2D matrices.

```v_kmult(v, k)```\
Multiply the whole vector to the real number given, it modifies the vector.\
Input: v = 1D vector, k = real number.

```m_kmult(m, k)```\
Multiply the whole matrix to the real number given, it modifies the matrix.\
Input: m = 2D matrix, k = real number.

```v_kpow(v, k)```

```m_kpow(m, k)```

```dotmm(m1, m2)```\
Dot product between two matrices (m1 x m2).\
Input: two 2D matrices.\
Output: 2D matrix.

```dotvm(v, m)```\
Dot product between a vector and a matrix (v x m), the order CANNOT be changed.\
Input: v = 1D vector, m = 2D matrix.\
Output: 1D vector.

```dotvv(v1, v2)```\
Dot product between two vectors (v1 x v2).\
Input: two 1D vectors.\
Output: real number.

```transpose(x)```\
Transpose a 2D matrix modifying it.\
Input: 2D matrix.

## License
```
MIT License

Copyright (c) 2020 Davide Fassio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Authors
Davide Fassio ([@Davidefassio](https://github.com/Davidefassio)).
