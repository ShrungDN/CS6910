sweep_configuration_1 = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [5, 10]},
        'n_hidden': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'lr': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_initialization': {'values': ['random', 'Xavier']},
        'activation': {'values': ['identity', 'sigmoid', 'ReLU', 'tanh']},
        'loss_func': {'values': ['cross_entropy', 'mean_squared_error']},
        'momentum': {'values': [0.9]},
        'beta': {'values': [0.9]},
        'beta1': {'values': [0.9]},
        'beta2': {'values': [0.999]},
        'epsilon': {'values': [1e-6]},
        'data_augmentation': {'values': [False]},
        'data_scaling': {'values': ['min_max', 'standard']}
    }
}