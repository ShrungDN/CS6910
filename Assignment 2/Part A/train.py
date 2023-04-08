from helper_functions import *
from model import CNNModel
from parse_args import parse_arguments

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta, Adagrad, Adam, NAdam, RMSprop

def main(config, train_data_path, test_data_path):
  IMGDIMS = config['IMGDIMS']
  NUM_WORKERS = config['NUM_WORKERS']
  BATCH_SIZE = config['BATCH_SIZE']
  MEAN, STD = config['MEAN_STD']
  DATA_AUG = config['DATA_AUG']
  LR = config['LR']
  EPOCHS = config['EPOCHS']
  OPTIM = config['OPTIM']
  LOSS_FUNC = config['LOSS_FUNC']

  train_transform, val_test_transform = get_transforms(DATA_AUG, IMGDIMS, MEAN, STD)

  train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
  valid_dataset = ImageFolder(root=test_data_path, transform=val_test_transform)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

  class_to_idx = train_dataset.class_to_idx
  classes = train_dataset.classes

  device = ('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Device: {device}\n")

  model = CNNModel()
  model.to(device, non_blocking=True)
  summary(model, (3, IMGDIMS[0], IMGDIMS[1]))

  optimizer = OPTIM(model.parameters(), lr=LR)
  criterion = LOSS_FUNC()

  train_loss, valid_loss = [], []
  train_acc, valid_acc = [], []
  for epoch in range(EPOCHS):
      print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
      train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion, device)
      valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                   criterion, device)
      train_loss.append(train_epoch_loss)
      valid_loss.append(valid_epoch_loss)
      train_acc.append(train_epoch_acc)
      valid_acc.append(valid_epoch_acc)
      print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
      print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
      print('-'*50)
  print('TRAINING COMPLETE')

if __name__ == '__main__':
#   args = parse_arguments()

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