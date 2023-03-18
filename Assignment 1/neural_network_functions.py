from helper_functions import *

def init_params(n_inp, n_hidden, n_out, init_type, seed=None):
  if seed is not None:
    np.random.seed(seed)
  NN = [n_inp] + n_hidden + [n_out]
  params = {}
  for i in range(1, len(NN)):
    if init_type == 'random':
      params['W' + str(i)] = np.random.randn(NN[i-1], NN[i]) * 0.01
    elif init_type == 'Xavier':
      params['W' + str(i)] = np.random.randn(NN[i-1], NN[i]) * np.sqrt(2 / (NN[i-1] + NN[i]))
    params['B' + str(i)] = np.zeros((1, NN[i]))
  return params

def forward(inp, params, activation):
  L = len(params) // 2
  cache = {'H0':inp}
  for i in range(1, L+1):
    cache['A' + str(i)] = cache['H' + str(i-1)] @ params['W' + str(i)] + params['B' + str(i)]
    cache['H' + str(i)] = activation(cache['A' + str(i)])
  cache['A' + str(L)] = cache['H' + str(L-1)] @ params['W' + str(L)] + params['B' + str(L)]
  cache['H' + str(L)] = softmax(cache['A' + str(L)])
  yhat = cache['H' + str(L)]
  return yhat, cache

def eval_params(x, y, params, config):
  act, act_der = get_activation(config['activation'])
  loss_func, loss_func_der = get_loss_func(config['loss_func'])
  WD = config['WD']
  yhat, _ = forward(x, params, act)
  loss = loss_func(y, yhat) + np.sum([0.5 * WD * np.linalg.norm(params[k]) for k in params.keys()])
  acc = accuracy(y, yhat)
  return loss, acc

def backward(y, params, yhat, cache, act_der, loss_func_der, WD):
  L = len(params) // 2
  m = y.shape[0]
  
  del_params = {} 
  del_ak = loss_func_der(y, yhat)
  for k in range(L, 0, -1):
    del_params['W' + str(k)] = ((cache['H' + str(k-1)].T @ del_ak) + WD * params['W' + str(k)]) / m
    del_params['B' + str(k)] = np.sum(del_ak, axis=0, keepdims=True) / m
    if k != 1:
      del_ak = (del_ak @ params['W' + str(k)].T) * act_der(cache['A' + str(k-1)])
  return del_params

