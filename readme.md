# General and Modular framework for training and Analyzing neural networks

## Structure

    main - loads a base config file, sets the seed(thus sneed to specify run config somewhere), builds model, optimizer, scheduler, dataloaders, and loggers, then calls the trainer to train the model. 

    train - contains Trainer class, which handles everything involving training, where the model, loss function, optimizer, and diagnostics are "abstracted away". 

    logger - logs EVERYTHING we want to save throughout our runs, including plots metrics versions, seed config, hyperparameters and diagnostics. Integrates with wandb and should save stuff in a runs folder. 

    utils/setup - methods that main class to build a specific optimizer scheduler, model. Need to make it more abstract and depend on the config. 

    utils/sweeper - not sure if will use. It's supposed to contain methods to run from sweep.py. You use a sweep config file to specify what you want to sweep over, allows for statistics over multiple runs. High chance of error here. Not sure what it should dsave. 

    runs/ - where run csvs and whatever relevant data wil be saved. 
    
    models - contains the files for each model we wish to use. 

    diagnostics - should contain a list of methods (with same structure) that are evaluated once every couple epochs. Stuff like latent graphs, reconstructions, and jacobians. Stuff you don't want to compute every epoch. Which ones you want to use defined in config file, and this is also called from the logger. Should be automatic. 


    configs - the meat and potatoes - should contain EVERYTHING we might want to customize, will need to be careful these are right. Set up custom details for hyperparameters, optimizers, models, data, and everything in between here. 

    sweep_configs - the configs specifically to be used for running sweeps, ie runs over many differen thyperparameters, or for larget statistical ensemble runs. 


    Analyze - methods which are supposed to do statistics on the logs of completed runs. 

    data - defines the data loaders - will depend on what structure the data has. 





