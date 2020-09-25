/*
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
*/

class NeuralNetwork{
    // NEURAL NETWORK ##########################################################

    // Set up the class.
    constructor(){
        this.n_layers = 0;
        this.layers = [];
        this.neurons = [];
        this.weights = [];
        this.activation_funcs = [];
        this.learning_rate = 0.1; // Default value.
		this.loss_func = "mse";   // Default loss function.
    }

    // Add a layer of neurons to the model.
    add_layer(neurons, activ_func){
        // Integrity checks
        if(neurons <= 0){ return false; }
        if(!(activ_func === "linear" || activ_func === "relu" || activ_func === "softplus" || activ_func === "sigmoid" || activ_func === "tanh" || activ_func === "softmax")){ return false; }
        // Add the layer
        this.layers.push(neurons);
        this.neurons.push([[]]);
        if(this.n_layers != 0){ // Initialize the weights using "He initalization" for ReLU/softplus layers, "Xavier initalization" otherwise.
            if(activ_func === "relu" || activ_func === "softplus"){ this.weights.push(this.rand(this.layers[this.n_layers-1]+1, this.layers[this.n_layers], 0, 2 / this.layers[this.n_layers-1])); }  // ReLU, softplus.
            else{ this.weights.push(this.rand(this.layers[this.n_layers-1]+1, this.layers[this.n_layers], 0, 1 / this.layers[this.n_layers-1])); }  // Linear, sigmoid, tanh, softmax.
        }
        this.n_layers++;
        this.activation_funcs.push(activ_func);
		return true;
    }

    // Set the learning rate if it's greater than 0.
    set_learning_rate(l){
        if(l > 0){
            this.learning_rate = l;
            return true;
        }
        return false;
    }

	// Set the loss (cost) function of the model if it is valid.
	// Possible loss functions:
	// 		- "mse" (mean squared error);
	// 		- "log_loss) (cross entropy loss).
	set_loss_function(f){
		if(f === "mse" || f === "log_loss"){
			this.loss_func = f;
			return true;
		}
		return false;
	}

    // Guess and return an output given the input.
    guess(input){
        this.neurons[0][0] = input; // Set up the first layer
        this.neurons[0][0] = this.activate(this.neurons[0][0], 0); // Activate the first layer
        for(let i = 1; i < this.n_layers; ++i){ // Propagate forward
            this.neurons[i-1][0].push(1); // Bias
            this.neurons[i][0] = this.dotvm(this.neurons[i-1][0], this.weights[i-1]); // Dot product
            this.neurons[i][0] = this.activate(this.neurons[i][0], i); // Apply the activation function
        }
        return this.neurons[this.n_layers-1][0]; // Return the output
    }

    // TODO
    backprop(){

    }

	// LOSS FUNCTIONS ##########################################################

	// Mean squared error.
	mse(y_err, y_cor){
		if(y_err.length !== y_cor.length){ return false; }
		let c = Math.pow(y_err[0] - y_cor[0], 2), l = y_err.length;
		for(let i = 1; i < l; ++i){
			c += Math.pow(y_err[i] - y_cor[i], 2);
		}
		return c / l;
	}

	// Binary cross entropy loss.
	log_loss(y_err, y_cor){
		if(y_err.length !== y_cor.length){ return false; }
		let c = (y_cor[0] === 1) ? Math.log(y_err[0]) : Math.log(1 - y_err[0]), l = y_err.length;
		for(let i = 1; i < l; ++i){
			c += (y_cor[i] === 1) ? Math.log(y_err[i]) : Math.log(1 - y_err[i]);
		}
		return - c / l;
	}

    // ACTIVATION FUNCTIONS ####################################################

    // Apply the correct activation function to the array passed.
    activate(x, i){
		if(this.activation_funcs[i] === "linear"  ){ return this.linear(x);   }
        if(this.activation_funcs[i] === "relu"    ){ return this.relu(x);     }
		if(this.activation_funcs[i] === "softplus"){ return this.softplus(x); }
    	if(this.activation_funcs[i] === "sigmoid" ){ return this.sigmoid(x);  }
		if(this.activation_funcs[i] === "tanh"    ){ return this.tanh(x);     }
        if(this.activation_funcs[i] === "softmax" ){ return this.softmax(x);  }
		return false;
    }

	linear(x){
		return Array.from(x);
	}

    relu(x){
		return Array.from(x, a => Math.max(0, a));
    }

    softplus(x){
		return Array.from(x, a => Math.log1p(Math.exp(a)));
    }

    sigmoid(x){ // Implemented in the numerical stable way
		return Array.from(x, a => (a >= 0) ? 1 / (1 + Math.exp(-a)) : Math.exp(a) / (1 + Math.exp(a)));
    }

    tanh(x){
		return Array.from(x, a => Math.tanh(a));
    }

    softmax(x){ // Implemented in the numerical stable way
		let l = x.length, v = new Array(l), exp_sum = 0;
        for(let i = 0; i < l; ++i){
            v[i] = x[i] - Math.max(...x);
            exp_sum += Math.exp(v[i]);
        }
        for(let i = 0; i < l; ++i){
            v[i] = Math.exp(v[i]) / exp_sum;
        }
		return v;
    }

    // The derivatives expect the vector passed to be activated using
    // the same activation function.
	der_linear(x){
		return Array.from(x, a => 1);
	}

    der_relu(x){
		return Array.from(x, a => (a === 0) ? 0 : 1);
    }

    der_softplus(x){
		return Array.from(x, a => 1 - 1/Math.exp(a));
    }

    der_sigmoid(x){
		return Array.from(x, a => a*(1 - a));
    }

    der_tanh(x){
		return Array.from(x, a => 1 - a*a);
    }

    der_softmax(x){
        let l = x.length, m = new Array(l);
        for(let i = 0; i < l; ++i){ m[i] = new Array(l); }
        for(let i = 0; i < l; ++i){
            for(let j = i; j < l; ++j){
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
    norm_distr(mean, stdev){
        let y1, x1, x2, w;
        do{
            x1 = 2 * Math.random() - 1;
            x2 = 2 * Math.random() - 1;
            w  = x1 * x1 + x2 * x2;
        } while( w >= 1);
        return mean + stdev * x1 * Math.sqrt((-2 * Math.log(w))/w);
    }

    // LINEAR ALGEBRA ##########################################################

	// Sum all elements of a 1D array.
	v_sumAll(v){
		return v.reduce((a, b) => a + b, 0);
	}

	// Sum all elements of a 2D matrix.
	m_sumAll(m){
		return m.reduce((a, b) => a + b.reduce((x, y) => x + y, 0), 0);
	}

    // Sum two 1D vector and return a 1D vector.
    v_sum(v1, v2){
        if(v1.length !== v2.length){ return false; }
        let v = Array.from(v1);
        for(let i = 0; i < v.length; ++i){
            v[i] += v2[i];
        }
        return v;
    }

    // Sum two 2D matrix and return a 2D matrix.
    m_sum(m1, m2){
        if(m1.length !== m2.length || m1[0].length !== m2[0].length){ return; }
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
    v_sumTo(v, v2){
        if(v.length !== v2.length){ return false; }
        for(let i = 0; i < v.length; ++i){
            v[i] += v2[i];
        }
		return v;
    }

    // Sum the second matrix to the first one modifying it.
    m_sumTo(m, m2){
        if(m.length !== m2.length || m[0].length !== m2[0].length){ return; }
        for(let i = 0; i < m.length; ++i){
            for(let j = 0; j < m[0].length; ++j){
                m[i][j] += m2[i][j];
            }
        }
		return m;
    }

    // Subtract two 1D vector and return a 1D vector.
    v_sub(v1, v2){
        if(v1.length !== v2.length){ return false; }
        let v = Array.from(v1);
        for(let i = 0; i < v1.length; ++i){
            v[i] -= v2[i];
        }
        return v;
    }

    // Subtract two 2D matrix and return a 2D matrix.
    m_sub(m1, m2){
        if(m1.length !== m2.length || m1[0].length !== m2[0].length){ return; }
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
    v_subTo(v, v2){
        if(v.length !== v2.length){ return false; }
        for(let i = 0; i < v.length; ++i){
            v[i] -= v2[i];
        }
		return v;
    }

    // Subtract the second matrix to the first one modifying it.
    m_subTo(m, m2){
        if(m.length !== m2.length || m[0].length !== m2[0].length){ return; }
        for(let i = 0; i < m.length; ++i){
            for(let j = 0; j < m[0].length; ++j){
                m[i][j] -= m2[i][j];
            }
        }
		return m;
    }

    // Multiply all elements of a 1D vector by a real number k.
    v_kmult(v, k){
        for(let i = 0; i < v.length; ++i){ v[i] *= k; }
		return v;
    }

    // Multiply all elements of a 2D matrix by a real number k.
    m_kmult(m, k){
        for(let i = 0; i < m.length; ++i){
            for(let j = 0; j < m[0].length; ++j){
                m[i][j] *= k;
            }
        }
		return m;
    }

	// Elevate all elements of a 1D vector to a real number k.
    v_kpow(v, k){
        for(let i = 0; i < v.length; ++i){ v[i] = Math.pow(v[i], k); }
		return v;
    }

    // Elevate all elements of a 2D matrix to a real number k.
    m_kpow(m, k){
        for(let i = 0; i < m.length; ++i){
            for(let j = 0; j < m[0].length; ++j){
                m[i][j] = Math.pow(m[i][j], k);
            }
        }
		return m;
    }

    // Dot product between two 2D matrices.
    // #cols m1 MUST BE EQUAL TO #rows m2.
    // Return a 2D matrix (#rows m1 x #cols m2).
    dotmm(m1, m2){
        if(m1[0].length !== m2.length){ return false; }
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
        if(v.length !== m.length){ return false; }
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
        if(v1.length !== v2.length){ return false; }
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
}
