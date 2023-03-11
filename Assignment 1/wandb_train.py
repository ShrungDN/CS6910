import wandb
from train import *
from sweep_configurations import *

CONFIG = sweep_configuration_1
DATASET = 'fashion_mnist'

(xtrain_inp, ytrain_inp), (xval_inp, yval_inp), (xtest_inp, ytest_inp), class_labels = get_dataset(DATASET)

wandb.login()
sweep_id = wandb.sweep(sweep=CONFIG, project='test-assgn')

def main():
  run = wandb.init()

  config = {'EPOCHS': wandb.config.epochs,
            'BATCH_SIZE': wandb.config.batch_size,
            'loss_func': 'cross_entropy',
            'optim': wandb.config.optimizer,
            'LR': wandb.config.lr,
            'MOMENTUM': 0.9,
            'BETA': 0.9,
            'BETA1': 0.9,
            'BETA2': 0.999, 
            'EPSILON': 0.000001,
            'WD': wandb.config.weight_decay,
            'W_init': wandb.config.weight_initialization,
            'activation': wandb.config.activation,
            'n_hidden': [wandb.config.hidden_size] * wandb.config.n_hidden 
            }

  str1 = f'ep:{wandb.config.epochs}_hl:{wandb.config.n_hidden}_nhl:{wandb.config.hidden_size}_l2:{wandb.config.weight_decay}_'
  str2 = f'lr:{wandb.config.lr}_opt:{wandb.config.optimizer}_bs:{wandb.config.batch_size}_wi:{wandb.config.weight_initialization}_act:{wandb.config.activation}'
  run.name = str1 + str2 

  params, logs = train(xtrain=xtrain_inp, ytrain=ytrain_inp, xval=xval_inp, yval=yval_inp, config=config, verbose=True, seed=0)
  
  for i in range(len(logs['epochs'])):
    wandb.log({
        'epochs': logs['epochs'][i],
        'train_acc': logs['train_acc'][i],
        'train_loss': logs['train_loss'][i], 
        'val_acc': logs['val_acc'][i], 
        'val_loss': logs['val_loss'][i]
    })

wandb.agent(sweep_id, function=main, count=100)