# CS6910
Shrung D N - ME19B168 - Assignment 1

**Usage**
```
usage: python3 train.py [-h --help HELP] 
                        [-wp --wandb_project]
                        [-we --wandb_entity]
                        [-wn --wandb_name]
                        [-wl --wandb_log]
                        [-d --dataset]
                        [-e --epochs]
                        [-b --batch_size]
                        [-l --loss]
                        [-o --optimizer]
                        [-lr --learning_rate]
                        [-m --momentum]
                        [-beta --beta]
                        [-beta1 --beta1]
                        [-beta2 --beta2]
                        [-eps --epsilon]
                        [-w_d --weight_decay]
                        [-w_i --weight_init]
                        [-nhl --num_layers]
                        [-sz --hidden_size]
                        [-a --activation]
                        [-ds --data_scaling]             	
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