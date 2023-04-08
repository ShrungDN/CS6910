from helper_functions import *
from model import CNNModel
from parse_args import parse_arguments

from torchsummary import summary

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
    print(f"Training: Epoch {epoch+1} / {EPOCHS}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
    val_epoch_loss, val_epoch_acc = validate(model, val_loader, criterion, device)
    
    logs['train_loss'].append(train_epoch_loss)
    logs['train_acc'].append(train_epoch_acc)
    logs['val_loss'].append(val_epoch_loss)
    logs['val_acc'].append(val_epoch_loss)

    print(f'Training: Loss = {train_epoch_loss:.4f} Accuracy = {train_epoch_acc:.4f}  Validation: Loss = {val_epoch_loss:.4f} Accuracy = {val_epoch_acc:.4f}')
    print('-'*50)

  model_metrics = eval_model(model, train_loader, val_loader, test_loader, criterion, device)
  return model, logs, model_metrics

if __name__ == '__main__':
  args = parse_arguments()

  config = {
    'IMGDIMS': (256, 256),
    'BATCH_SIZE': 64,
    'MEAN_STD': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'DATA_AUG': True,
    'LR': 1e-3,
    'EPOCHS': 5,
    'OPTIM': Adadelta,
    'LOSS_FUNC': CrossEntropyLoss
  }

  config = {'IMGDIMS': (args.dimsw, args.dimsh),
            'BATCH_SIZE': args.batch_size,
            'MEAN_STD': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            'DATA_AUG': args.data_aug,
            'LR': args.learning_rate,
            'EPOCHS': args.epochs,
            'OPTIM': get_optimizer(args.optimizer),
            'LOSS_FUNC': get_loss_func(args.loss)
            }
  
  
  # main(config, '/content/inaturalist_12K/train', '/content/inaturalist_12K/val')
  model, logs, model_metrics = main(config, args.train_data_path, args.test_data_path)

  print('Final Model Metrics:')
  print('Training: Accuracy = {} Loss = {}'.format(model_metrics['train_acc'], model_metrics['train_loss']))
  print('Validation: Accuracy = {} Loss = {}'.format(model_metrics['val_acc'], model_metrics['val_loss']))
  print('Testing: Accuracy = {} Loss = {}'.format(model_metrics['test_acc'], model_metrics['test_loss']))
  print()

  if args.wandb_log == 'True':
    import wandb
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    NAME = args.wandb_name

    wandb.login()
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    wandb.log(config)

    for i in range(len(logs['epochs'])):
      wandb.log({
          'epochs': logs['epochs'][i],
          'train_acc': logs['train_acc'][i],
          'train_loss': logs['train_loss'][i], 
          'val_acc': logs['val_acc'][i], 
          'val_loss': logs['val_loss'][i]
      })

    wandb.log({'Train Accuracy': model_metrics['train_acc']})
    wandb.log({'Validation Accuracy': model_metrics['val_acc']})
    wandb.log({'Test Accuracy': model_metrics['test_acc']})
    
    wandb.finish()    