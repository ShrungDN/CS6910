sweep_configuration_1 = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'n_hidden': {'values': [1, 5]},
        'hidden_size': {'values': [32, 64]},
        'weight_decay': {'values': [0, 0.1]},
        'lr': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [64]},
        'weight_initialization': {'values': ['random', 'Xavier']},
        'activation': {'values': ['identity', 'sigmoid', 'ReLU', 'tanh']},
        'loss_func': {'values': ['cross_entropy']},
        'momentum': {'values': [0.9]},
        'beta': {'values': [0.9]},
        'beta1': {'values': [0.9]},
        'beta2': {'values': [0.999]},
        'epsilon': {'values': [1e-6]},
        'data_scaling': {'values': ['standard']}
    }
}


sweep_configuration_2 = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [20]},
        'n_hidden': {'values': [1, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.1]},
        'lr': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['momentum','rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [64]},
        'weight_initialization': {'values': ['Xavier']},
        'activation': {'values': ['ReLU', 'tanh']},
        'loss_func': {'values': ['cross_entropy']},
        'momentum': {'values': [0.9]},
        'beta': {'values': [0.9]},
        'beta1': {'values': [0.9]},
        'beta2': {'values': [0.999]},
        'epsilon': {'values': [1e-6]},
        'data_scaling': {'values': ['standard']}
    }
}

sweep_configuration_3 = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [20, 30]},
        'n_hidden': {'values': [5]},
        'hidden_size': {'values': [128]},
        'weight_decay': {'values': [0.1, 0.2]},
        'lr': {'values': [1e-4, 1e-5]},
        'optimizer': {'values': ['adam', 'nadam']},
        'batch_size': {'values': [64]},
        'weight_initialization': {'values': ['Xavier']},
        'activation': {'values': ['ReLU']},
        'loss_func': {'values': ['cross_entropy']},
        'momentum': {'values': [0.9]},
        'beta': {'values': [0.9]},
        'beta1': {'values': [0.9, 0.8]},
        'beta2': {'values': [0.999, 0.99]},
        'epsilon': {'values': [1e-6]},
        'data_scaling': {'values': ['standard']}
    }
}