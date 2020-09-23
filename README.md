# nnrl (Neural network with reinforcement learning)
![alt text](logo.png)

Neural network and reinforcement learning implementation in JS.\
Developing phase.

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

```guess(input)```\
Return a guess based on the input vector given using forward propagation.\
Input: 1D vector.\
Output: 1D vector.

```backprop``` TODO

### Activation functions
#### A bunch of activation functions (funny stuff here), their derivatives and the functions to apply them.
```activate(x, i)```\
Apply the activation function to the layer.\
Input: x = the vector containing the layer to modify, i = the number of the layer (0 = input layer, ...).

#### All the activation functions take as input a vector and modify it.
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

#### All the derivatives of the activation functions take as input a vector (ALREADY ACTIVATED with the same activation function) and return another vector as result.
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
Input: 1D vector. Output: 1D vector.

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
```vec_sum(v1, v2)```\
Sum two vectors without modifying them and return a vector containing the result.\
Input: two 1D vectors.\
Output: 1D vector.

```mat_sum(m1, m2)```\
Sum two matrices without modifying them and return a matrix containing the result.\
Input: two 2D matrices.\
Output: 2D matrix.

```vec_sumTo(v, v2)```\
Sum the second vector to the first one, the first vector is therefore modified.\
Input: two 1D vectors.

```mat_sumTo(m, m2)```\
Sum the second matrix to the first one, the first matrix is therefore modified.\
Input: two 2D matrices.

```vec_sub(v1, v2)```\
Subtract two vectors without modifying them and return a vector containing the result.\
Input: two 1D vectors.\
Output: 1D vector.

```mat_sub(m1, m2)```\
Subtract two matrices without modifying them and return a matrix containing the result.\
Input: two 2D matrices.\
Output: 2D matrix.

```vec_subTo(v, v2)```\
Subtarct the second vector to the first one, the first vector is therefore modified.\
Input: two 1D vectors.

```mat_subTo(m, m2)```\
Subtract the second matrix to the first one, the first matrix is therefore modified.\
Input: two 2D matrices.

```vec_kmult(k, v)```\
Multiply the whole vector to the real number given, it modifies the vector.\
Input: k = real number, v = 1D vector.

```mat_kmult(k, m)```\
Multiply the whole matrix to the real number given, it modifies the matrix.\
Input: k = real number, m = 2D matrix.

```dotmm(m1, m2)```\
Dot product between two matrices (m1 x m2).\
Input: two 2D matrices.\
Output: 2D matrix.

```dotvm(v, m)```\
Dot product between a vector and a matrix (v x m).\
Input: v = 1D vector, m = 2D matrix.\
Output: 1D vector.

```dotvv(v1, v2)```\
Dot product between two vectors (v1 x v2).\
Input: two 1D vectors.\
Output: real number.

```transpose(x)```\
Transpose a 2D matrix modifying it.\
Input: 2D matrix.
