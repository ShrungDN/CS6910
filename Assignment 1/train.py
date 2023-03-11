from neural_network_functions import *

def train(xtrain, ytrain, xval, yval, config, verbose=False, seed=None):

  EPOCHS = config['EPOCHS']
  BATCH_SIZE = config['BATCH_SIZE']
  lfunc = config['loss_func']
  optim = config['optim']
  LR = config['LR']
  MOMENTUM = config['MOMENTUM']
  BETA = config['BETA']
  BETA1 = config['BETA1']
  BETA2 = config['BETA2']
  EPSILON = config['EPSILON']
  WD = config['WD']
  W_init = config['W_init']
  activation = config['activation']
  n_hidden = config['n_hidden']
  
  if seed is not None:
    np.random.seed(0)
  params = init_params(n_inp = xtrain.shape[1], n_hidden = n_hidden, n_out = ytrain.shape[1], init_type= W_init, seed = seed)

  # if optim=='adam':
  #   for k in params:
  #     params[k] = params[k].astype(np.float128)

  act, act_der = get_activation(activation)
  loss_func, loss_func_der = get_loss_func(lfunc)

  U = {k:0 for k in params.keys()}
  V = {k:0 for k in params.keys()}

  logs = {
      'epochs': [],
      'train_loss': [],
      'train_acc': [],
      'val_loss': [],
      'val_acc': []
  }

  for i in range(1, EPOCHS+1):

    temp = np.arange(xtrain.shape[0])
    np.random.shuffle(temp)
    n_batches = xtrain.shape[0] // BATCH_SIZE
    batches = temp[:n_batches*BATCH_SIZE].reshape(-1, BATCH_SIZE).tolist()
    if  xtrain.shape[0] % BATCH_SIZE != 0:
      batches = batches + [temp[n_batches*BATCH_SIZE:].tolist()]

    for j, batch in enumerate(batches):
      if optim == 'sgd':
        yhat, cache = forward(xtrain[batch, :], params, act)
        del_params = backward(ytrain[batch], params, yhat, cache, act_der, loss_func_der, WD) 
        for k in params.keys():
          params[k] = params[k] - LR * del_params[k]

      elif optim == 'momentum':
        yhat, cache = forward(xtrain[batch, :], params, act)
        del_params = backward(ytrain[batch], params, yhat, cache, act_der, loss_func_der, WD) 
        for k in params.keys():
          U[k] = MOMENTUM * U[k] + del_params[k]
          params[k] = params[k] - LR * U[k]

      elif optim == 'nag':
        params_LA = {k: (params[k] - MOMENTUM * U[k]) for k in params.keys()}
        yhat, cache = forward(xtrain[batch, :], params_LA, act)
        del_params = backward(ytrain[batch], params_LA, yhat, cache, act_der, loss_func_der, WD) 
        for k in params.keys():
          U[k] = MOMENTUM * U[k] + del_params[k]
          params[k] = params[k] - LR * U[k]

      elif optim == 'rmsprop':
        yhat, cache = forward(xtrain[batch, :], params, act)
        del_params = backward(ytrain[batch], params, yhat, cache, act_der, loss_func_der, WD) 
        for k in params.keys():
          V[k] = BETA * V[k] + (1-BETA) * (del_params[k] ** 2)
          params[k] = params[k] - (LR / np.sqrt(V[k] + EPSILON)) * del_params[k]

      elif optim=='adam':
        yhat, cache = forward(xtrain[batch, :], params, act)
        del_params = backward(ytrain[batch], params, yhat, cache, act_der, loss_func_der, WD) 
        n_updates = (i-1) * len(batches) + (j+1)
        for k in params.keys():
          U[k] = BETA1 * U[k] + (1-BETA1) * del_params[k]
          Uk_hat = U[k] / (1-BETA1**n_updates)
          V[k] = BETA2 * V[k] + (1-BETA2) * del_params[k]**2
          Vk_hat = V[k] / (1-BETA2**n_updates)
          params[k] = params[k] - (LR / (np.sqrt(Vk_hat) + EPSILON)) * Uk_hat

      elif optim=='nadam':
        yhat, cache = forward(xtrain[batch, :], params, act)
        del_params = backward(ytrain[batch], params, yhat, cache, act_der, loss_func_der, WD) 
        n_updates = (i-1) * len(batches) + (j+1)
        for k in params.keys():
          U[k] = BETA1 * U[k] + (1-BETA1) * del_params[k]
          Uk_hat = U[k] / (1-BETA1**n_updates)
          V[k] = BETA2 * V[k] + (1-BETA2) * del_params[k]**2
          Vk_hat = V[k] / (1-BETA2**n_updates)
          params[k] = params[k] - (LR / (np.sqrt(Vk_hat) + EPSILON)) * (BETA1 * Uk_hat + (1 - BETA1) * del_params[k] / (1 - BETA1**n_updates))
          
      else:
        raise Exception('Incorrect Optimizer')

    train_loss, train_acc = eval_params(xtrain, ytrain, params, act, loss_func, WD)
    val_loss, val_acc = eval_params(xval, yval, params, act, loss_func, WD)

    logs['epochs'].append(i)
    logs['train_loss'].append(train_loss)
    logs['train_acc'].append(train_acc)
    logs['val_loss'].append(val_loss)
    logs['val_acc'].append(val_acc)

    if verbose and (i % 1 == 0):
      print(f'Epoch {i}:: Training: Loss = {np.mean(train_loss):.4f} Accuracy = {train_acc:.4f}  Validation: Loss = {val_loss:.4f} Accuracy = {val_acc:.4f}')
      
  return params, logs

if __name__ == '__main__':
  # add parser and do training

    # config = {'EPOCHS': 2,
    #         'BATCH_SIZE': 64,
    #         'loss_func': 'cross_entropy',
    #         'optim': 'momentum',
    #         'LR': 1e-2,
    #         'MOMENTUM': 0.9,
    #         'BETA': 0.9,
    #         'BETA1': 0.9,
    #         'BETA2': 0.999, 
    #         'EPSILON': 0.000001,
    #         'WD': 0,
    #         'W_init': 'Xavier',
    #         'activation': 'sigmoid',
    #         'n_hidden': [64, 64, 64]
    #         }

    # params, logs = train(xtrain=xtrain_inp, ytrain=ytrain_inp, xval=xval_inp, yval=yval_inp, config=config, verbose=True, seed=0)

    pass