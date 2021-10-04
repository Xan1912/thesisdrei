import wandb

sweep_config = {
    'method': 'grid', #grid, random
    'metric': {
      'name': 'loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'epochs': {
            'values': [2, 3, 4, 5, 6, 7, 8]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'train_set_size': {
            'values': [0.7]
        },
        'lr': {
            'values': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        },
        'eps':{
            'values':[1e-8,2e-8,3e-8]
        },
        'seed_val': {
            'values': [42]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="thesis")