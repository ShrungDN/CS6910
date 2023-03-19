# CS6910
Shrung D N - ME19B168 - Assignment 1

**Usage**
```
usage: python3 train.py [-h --help HELP] 
                        [-wp --wandb_project] <string>
                        [-we --wandb_entity] <string>
                        [-wn --wandb_name] <string>
                        [-wl --wandb_log] "True" or "False"
                        [-d --dataset] <string>
                        [-e --epochs] <int>
                        [-b --batch_size] <int>
                        [-l --loss] "cross_entropy", "mean_squared_error"
                        [-o --optimizer] "sgd", "momentum", "nag", "rmsprop", "adam", "nadam"
                        [-lr --learning_rate] <float>
                        [-m --momentum] <float>
                        [-beta --beta] <float>
                        [-beta1 --beta1] <float>
                        [-beta2 --beta2] <float>
                        [-eps --epsilon] <float>
                        [-w_d --weight_decay] <float>
                        [-w_i --weight_init] "random", "Xavier"
                        [-nhl --num_layers] <int>
                        [-sz --hidden_size] <int>
                        [-a --activation] "identity", "sigmoid", "tanh", "ReLU"
                        [-ds --data_scaling] "min_max", "standard"       	
```

Optimal Hyperparameters found for Fashion MNIST dataset:

epochs: 30

batch_size: 128

loss: cross_entropy

optimizer: adam

learning_rate: 0.0001

beta1: 0.8

beta2: 0.999

epsilon: 0.000001

weight_decay: 0.3

weight_init: Xavier 

num_layers: 5

hidden_size: 128

activation: ReLU