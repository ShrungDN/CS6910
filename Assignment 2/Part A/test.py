import wandb

wandb.login()

config = {'a':1,
          'b':2,
          'c':3}

run = wandb.init(entity='me19b168', project='test-assgn', name='test')

wandb.log({'config':config})

for i in range(10):
    wandb.log({'i':i})

wandb.finish()