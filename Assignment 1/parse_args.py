import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-wp', '--wandb_project', type=str, default='ME19B168_CS6910_Assgn1', help='Project name on WandB')
    parser.add_argument('-we', '--wandb_entity', type=str, default='ME19B168', help='Username on WandB')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', help='Dataset to be used: "fashion_mnist" or "mnist"')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used for training')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', help='Loss function: "cross_entropy" or "mean_squared_error"')
    parser.add_argument('-o', '--optimizer', type=str, default='nadam', help='Optimizer to be used: "sgd", "momentum", "nag", "rmsprop", "adam" or "nadam"')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning Rate to be used')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.05, help='Weight decay used by optimizers.')
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier', help='Type of weight initialization: "random" or "Xavier"')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers in the neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=64, help='Number of neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, default='ReLU', help='Activation function to be used: "identity", "sigmoid", "tanh" or "ReLU')
    parser.add_argument('-ds', '--dataset_scaling', type=str, default='standard', help='Type of scaling to be used for the data: "min_max" or "standard"')
   
    args = parser.parse_args()
    return args