from helper_functions import *
from model import CNNModel
from parse_args import parse_arguments

from torchsummary import summary
from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta, Adagrad, Adam, NAdam, RMSprop

def main(config, train_data_path, test_data_path):
  IMGDIMS = config['IMGDIMS']
  BATCH_SIZE = config['BATCH_SIZE']
  MEAN, STD = config['MEAN_STD']
  DATA_AUG = config['DATA_AUG']
  LR = config['LR']
  EPOCHS = config['EPOCHS']
  OPTIM = config['OPTIM']
  LOSS_FUNC = config['LOSS_FUNC']

  device = ('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Device: {device}\n")

  train_transform, val_test_transform = get_transforms(DATA_AUG, IMGDIMS, MEAN, STD)

  train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(train_data_path, train_transform, test_data_path, val_test_transform, BATCH_SIZE)

  model = CNNModel()
  model.to(device, non_blocking=True)
  summary(model, (3, IMGDIMS[0], IMGDIMS[1]))
  print()

  optimizer = OPTIM(model.parameters(), lr=LR)
  criterion = LOSS_FUNC()

  logs = {
     'epochs': [],
     'train_loss': [],
     'train_acc': [],
     'val_loss': [],
     'val_acc': []
  }

  for epoch in range(EPOCHS):
      train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
      val_epoch_loss, val_epoch_acc = validate(model, val_loader, criterion, device)
      
      logs['train_loss'].append(train_epoch_loss)
      logs['train_acc'].append(train_epoch_acc)
      logs['val_loss'].append(val_epoch_loss)
      logs['val_acc'].append(val_epoch_loss)

      print(f"Training: Epoch {epoch+1} / {EPOCHS}")
      print(f'Training: Loss = {train_epoch_loss:.4f} Accuracy = {train_epoch_acc:.4f}  Validation: Loss = {val_epoch_loss:.4f} Accuracy = {val_epoch_acc:.4f}')
      print('-'*50)

  model_metrics = eval_model(train_loader, val_loader, test_loader, criterion, device)
  print('Final Model Metrics:')
  print('Training: Accuracy = {} Loss = {}'.format(model_metrics['train_acc'], model_metrics['train_loss']))
  print('Validation: Accuracy = {} Loss = {}'.format(model_metrics['val_acc'], model_metrics['val_loss']))
  print('Testing: Accuracy = {} Loss = {}'.format(model_metrics['test_acc'], model_metrics['test_loss']))

  return model

if __name__ == '__main__':
  args = parse_arguments()

  config = {
    'IMGDIMS': (256, 256),
    'NUM_WORKERS': 2,
    'BATCH_SIZE': 64,
    'MEAN_STD': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'DATA_AUG': True,
    'LR': 1e-3,
    'EPOCHS': 4,
    'OPTIM': Adadelta,
    'LOSS_FUNC': CrossEntropyLoss
  }

#    config = {'EPOCHS': args.epochs,
#             'BATCH_SIZE': args.batch_size,
#             'loss_func': args.loss,
#             'optim': args.optimizer,
#             'LR': args.learning_rate,
#             'MOMENTUM': args.momentum,
#             'BETA': args.beta,
#             'BETA1': args.beta1,
#             'BETA2': args.beta2, 
#             'EPSILON': args.epsilon,
#             'WD': args.weight_decay,
#             'W_init': args.weight_init,
#             'activation': args.activation,
#             'n_hidden': [args.hidden_size] * args.num_layers
#             }
   
  main(config, '/content/inaturalist_12K/train', '/content/inaturalist_12K/val')