import wandb
from train import *
from sweep_configurations import *

CONFIG = sweep_configuration_1
DATASET = 'fashion_mnist'
ENTITY = 'me19b168'
PROJECT ='ME19B168_CS6910_Assgn1'
NAME = 'me19b168'



(xtrain, ytrain), (xval, yval), (xtest, ytest), class_labels = get_dataset(DATASET)

wandb.login()

# Change _i to view different set of images
_i = 5 
_idxs = {k:np.where(ytrain==k)[0][_i] for k in set(ytrain)}
run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)
for _k in _idxs.keys():
    wandb.log({f'{class_labels[_k]}':[wandb.Image(xtrain[_idxs[_k]])]})
wandb.finish()

sweep_id = wandb.sweep(sweep=CONFIG, project=PROJECT)

def main():
  run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)
  (xtrain_inp, ytrain_inp), (xval_inp, yval_inp), (xtest_inp, ytest_inp) = scale_dataset(xtrain, ytrain, xval, yval, xtest, ytest, wandb.config.data_scaling)

  config = {'EPOCHS': wandb.config.epochs,
            'BATCH_SIZE': wandb.config.batch_size,
            'loss_func': wandb.config.loss_func,
            'optim': wandb.config.optimizer,
            'LR': wandb.config.lr,
            'MOMENTUM': wandb.config.momentum,
            'BETA': wandb.config.beta,
            'BETA1': wandb.config.beta1,
            'BETA2': wandb.config.beta2, 
            'EPSILON': wandb.config.epsilon,
            'WD': wandb.config.weight_decay,
            'W_init': wandb.config.weight_initialization,
            'activation': wandb.config.activation,
            'n_hidden': [wandb.config.hidden_size] * wandb.config.n_hidden
            }

  name = 'ep:{}_hl:{}_lr:{}_opt:{}_bs:{}'.format(config['EPOCHS'], f'{wandb.config.hidden_size}^{wandb.config.n_hidden}',
                                                 config['LR'], config['optim'], config['BATCH_SIZE'])
  run.name = name

  params, logs = train(xtrain=xtrain_inp, ytrain=ytrain_inp, xval=xval_inp, yval=yval_inp, config=config, verbose=True, seed=0)
  
  for i in range(len(logs['epochs'])):
    wandb.log({
        'epochs': logs['epochs'][i],
        'train_acc': logs['train_acc'][i],
        'train_loss': logs['train_loss'][i], 
        'val_acc': logs['val_acc'][i], 
        'val_loss': logs['val_loss'][i]
    })
  
  wandb.finish()

wandb.agent(sweep_id, function=main, count=50)