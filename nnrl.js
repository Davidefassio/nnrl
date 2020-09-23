/*
 * Class methods recap.
 *
 * Neural network:
 *  - constructor
 *  - add_layer
 *  - set_learning_rate
 *  - guess
 *  - backprop  (TODO)
 *
 * Activation functions:
 *  - activate
 *  - relu
 *  - softplus
 *  - sigmoid
 *  - tanh
 *  - softmax
 *  - der_relu
 *  - der_softplus
 *  - der_sigmoid
 *  - der_tanh
 *  - der_softmax
 *
 * Random generators:
 *  - rand
 *  - norm_distr
 *
 * Linear algebra:
 *  - vec_sum
 *  - mat_sum
 *  - vec_sumTo
 *  - mat_sumTo
 *  - vec_sub
 *  - mat_sub
 *  - vec_subTo
 *  - mat_subTo
 *  - vec_kmult
 *  - mat_kmult
 *  - dotmm
 *  - dotvm
 *  - dotvv
 *  - transpose
 */

// Ancora da aggiungere tutta la parte sul RL.

class NeuralNetwork{
    // NEURAL NETWORK ##########################################################

    // Set up the class.
    constructor(){
        this.n_layers = 0;
        this.layers = [];
        this.neurons = [];
        this.weights = [];
        this.activation_funcs = [];
        this.learning_rate = 0.1; // Default value
    }

    // Add a layer of neurons to the model.
    add_layer(neurons, activ_func){
        // Integrity checks
        if(neurons <= 0){ return false; }
        if(!(activ_func == "linear" || activ_func == "relu" || activ_func == "softplus" || activ_func == "sigmoid" || activ_func == "softmax")){ return false; }
        // Add the layer
        this.layers.push(neurons);
        this.neurons.push([[]]);
        if(this.n_layers != 0){ // Initialize the weights using "He initalization" for ReLU/softplus layers, "Xavier initalization" otherwise.
            if(activ_func == "relu" || activ_func == "softplus"){ this.weights.push(this.rand(this.layers[this.n_layers-1]+1, this.layers[this.n_layers], 0, 2 / this.layers[this.n_layers-1])); }  // ReLU, softplus.
            else{ this.weights.push(this.rand(this.layers[this.n_layers-1]+1, this.layers[this.n_layers], 0, 1 / this.layers[this.n_layers-1])); }  // Linear, sigmoid, softmax.
        }
        this.n_layers++;
        this.activation_funcs.push(activ_func);
    }

    // Set the learning rate if it's greater than 0.
    set_learning_rate(l){
        if(l > 0){ 
            this.learning_rate = l; 
            return true;
        }
        return false;
    }

    // Guess and return an output given the input.
    guess(input){
        this.neurons[0][0] = input; // Set up the first layer
        this.activate(this.neurons[0][0], 0); // Activate the first layer
        for(let i = 1; i < this.n_layers; ++i){ // Propagate forward
            this.neurons[i-1][0].push(1); // Bias
            this.neurons[i][0] = this.dotvm(this.neurons[i-1][0], this.weights[i-1]); // Dot product
            this.activate(this.neurons[i][0], i); // Apply the activation function
        }
        return this.neurons[this.n_layers-1][0]; // Return the output
    }

    // TODO
    backprop(){

    }

    // ACTIVATION FUNCTIONS ####################################################

    // Apply the correct activation function to the array passed
    // modifying its values.
    activate(x, i){
        if(this.activation_funcs[i] == "relu"){ this.relu(x); }
        else if(this.activation_funcs[i] == "sigmoid"){ this.sigmoid(x); }
        else if(this.activation_funcs[i] == "softmax"){ this.softmax(x); }
    }

    // The activation function modify the array passed.
    relu(x){
        for(let i = 0; i < x.length; ++i){ x[i] = Math.max(0, x[i]); }
    }

    softplus(x){
        for(let i = 0; i < x.length; ++i){ x[i] = Math.log1p(Math.exp(x[i])); }
    }

    sigmoid(x){ // Implemented in the numerical stable way
        let e;
        for(let i = 0; i < x.length; ++i){
            if(x[i] >= 0){
                x[i] = 1 / (1 + Math.exp(-x[i]));
            }
            else{
                e = Math.exp(x[i]);
                x[i] = e / (1 + e);
            }
        }
    }

    tanh(x){
        for(let i = 0; i < x.length; ++i){ x[i] = Math.tanh(x[i]); }
    }

    softmax(x){ // Implemented in the numerical stable way
        let exp_sum = 0, max = Math.max(...x);
        for(let i = 0; i < x.length; ++i){
            x[i] -= max;
            exp_sum += Math.exp(x[i]);
        }
        for(let i = 0; i < x.length; ++i){
            x[i] = Math.exp(x[i]) / exp_sum;
        }
    }

    // The derivations of the activation functions NOT modify the array passed.
    // The derivatives expect the vector passed to be activated using
    // the same activation function.
    der_relu(x){
        let v = new Array(x.length);
        for(let i = 0; i < x.length; ++i){ (x[i] == 0) ? v[i] = 0 : v[i] = 1; }
        return v;
    }

    der_softplus(x){
        let v = new Array(x.length);
        for(let i = 0; i < x.length; ++i){ v[i] = 1 - 1/Math.exp(x[i]); }
        return v;
    }

    der_sigmoid(x){
        let v = new Array(x.length);
        for(let i = 0; i < x.length; ++i){ v[i] = x[i]*(1 - x[i]); }
        return v;
    }

    der_tanh(x){
        let v = new Array(x.length);
        for(let i = 0; i < x.length; ++i){ v[i] = 1 - x[i]*x[i]; }
        return v;
    }

    der_softmax(x){
        let m = new Array(x.length);
        for(let i = 0; i < x.length; ++i){ m[i] = new Array(x.length); }
        for(let i = 0; i < x.length; ++i){
            for(let j = i; j < x.length; ++j){
                if(i != j){ m[i][j] = m[j][i] = -x[i]*x[j]; }
                else{ m[i][j] = x[i]*(1-x[j]); }
            }
        }
        return m;
    }

    // RANDOM GENERATORS #######################################################

    // Generate a 2D matrix of random normal distrubuted values.
    rand(row, col, mean, variance){
        let m = new Array(row), stdev = Math.sqrt(variance);
        for(let i = 0; i < row; ++i){
            m[i] = new Array(col);
            for(let j = 0; j < col; ++j){
                m[i][j] = this.norm_distr(mean, stdev);
            }
        }
        return m;
    }

    // Generate a number that follow the normal distribution
    // with mean and standard deviation given.
    norm_distr(mean, stdev) {
        let y1, x1, x2, w;
        do{
            x1 = 2 * Math.random() - 1;
            x2 = 2 * Math.random() - 1;
            w  = x1 * x1 + x2 * x2;
        } while( w >= 1);
        return mean + stdev * x1 * Math.sqrt((-2 * Math.log(w))/w);
    }

    // LINEAR ALGEBRA ##########################################################

    // Sum two 1D vector and return a 1D vector.
    vec_sum(v1, v2){
        if(v1.length != v2.length){ return false; }
        let v = new Array(v1.length);
        for(let i = 0; i < v1.length; ++i){
            v[i] = v1[i] + v2[i];
        }
        return v;
    }

    // Sum two 2D matrix and return a 2D matrix.
    mat_sum(m1, m2){
        if(m1.length != m2.length || m1[0].length != m2[0].length){ return; }
        let m = new Array(m1.length);
        for(let i = 0; i < m1.length; ++i){
            m[i] = new Array(m1[0].length);
            for(let j = 0; j < m1[0].length; ++j){
                m[i][j] = m1[i][j] + m2[i][j];
            }
        }
        return m;
    }

    // Sum the second vector to the first one modifying it.
    vec_sumTo(v, v2){
        if(v.length != v2.length){ return false; }
        for(let i = 0; i < v.length; ++i){
            v[i] += v2[i];
        }
    }

    // Sum the second matrix to the first one modifying it.
    mat_sumTo(m, m2){
        if(m.length != m2.length || m[0].length != m2[0].length){ return; }
        for(let i = 0; i < m.length; ++i){
            for(let j = 0; j < m[0].length; ++j){
                m[i][j] += m2[i][j];
            }
        }
    }

    // Subtract two 1D vector and return a 1D vector.
    vec_sub(v1, v2){
        if(v1.length != v2.length){ return false; }
        let v = new Array(v1.length);
        for(let i = 0; i < v1.length; ++i){
            v[i] = v1[i] - v2[i];
        }
        return v;
    }

    // Subtract two 2D matrix and return a 2D matrix.
    mat_sub(m1, m2){
        if(m1.length != m2.length || m1[0].length != m2[0].length){ return; }
        let m = new Array(m1.length);
        for(let i = 0; i < m1.length; ++i){
            m[i] = new Array(m1[0].length);
            for(let j = 0; j < m1[0].length; ++j){
                m[i][j] = m1[i][j] - m2[i][j];
            }
        }
        return m;
    }

    // Subtract the second vector to the first one modifying it.
    vec_subTo(v, v2){
        if(v.length != v2.length){ return false; }
        for(let i = 0; i < v.length; ++i){
            v[i] -= v2[i];
        }
    }

    // Subtract the second matrix to the first one modifying it.
    mat_subTo(m, m2){
        if(m.length != m2.length || m[0].length != m2[0].length){ return; }
        for(let i = 0; i < m.length; ++i){
            for(let j = 0; j < m[0].length; ++j){
                m[i][j] -= m2[i][j];
            }
        }
    }

    // Multiply all elements of a 1D vector by a real number k.
    vec_kmult(k, v){
        for(let i = 0; i < v.length; ++i){
            v[i] *= k;
        }
    }

    // Multiply all elements of a 2D matrix by a real number k.
    mat_kmult(k, m){
        for(let i = 0; i < m.length; ++i){
            for(let j = 0; j < m[0].length; ++j){
                m[i][j] *= k;
            }
        }
    }

    // Dot product between two 2D matrices.
    // #cols m1 MUST BE EQUAL TO #rows m2.
    // Return a 2D matrix (#rows m1 x #cols m2).
    dotmm(m1, m2){
        if(m1[0].length != m2.length){ return false; }
        let r = m1.length, c = m2[0].length, n = m1[0].length, m = new Array(r);
        for(let i = 0; i < r; ++i){
            m[i] = new Array(c);
            for(let j = 0; j < c; ++j){
                m[i][j] = 0;
                for(let k = 0; k < n; ++k){
                    m[i][j] += (m1[i][k]*m2[k][j]);
                }
            }
        }
        return m;
    }

    // Dot product between a 1D vector and a 2D matrix.
    // #elements v MUST BE EQUAL TO #rows m.
    // Return a 1D vector (1 x #cols m).
    dotvm(v, m){
        if(v.length != m.length){ return false; }
        let c = m[0].length, n = v.length, a = new Array(c);
        for(let j = 0; j < c; ++j){
            a[j] = 0;
            for(let k = 0; k < n; ++k){
                a[j] += (v[k]*m[k][j]);
            }
        }
        return a;
    }

    // Dot product between two 1D vector.
    // #elements v1 MUST BE EQUAL TO #elements v2.
    // Return a real number.
    dotvv(v1, v2){
        if(v1.length != v2.length){ return false; }
        let n = 0;
        for(let i = 0; i < v1.length; ++i){
            n += v1[i]*v2[i];
        }
        return n;
    }

    // Transpose a 2D matrix.
    transpose(x){
        return x[0].map((_, colIndex) => x.map(row => row[colIndex]));
    }
