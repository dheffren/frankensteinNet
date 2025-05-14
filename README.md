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



## Config File Layout
    













## Outline of Program run Order
    Command Line Functions: 
    Main.py - Call ONE run of a model given a config file (called from sweep.py). If not called from sweeps, runs in runs/manualRuns. 
    Sweep.py - Runs a sweep based on sweep_configs, saves a config in .sweep_tmp. Runs in a separate folder (runs/sweeps)
    Manage.py - Delete and list runs. Doesn't work right now. Uses methods from RunManager. Supposed to allow to resume as well. 

    Now, let's go into the loop for main.py
    setup.py contains the "building" of the model, dataset, optimizers and schedulers based on the config. 
    data.py constructs the dataset from datasets folder - these datasets are AUTOMATICALLY "registered' based on name. All you need to do is just make a new dataset class in datasets/*, and pass in its name in the config. 
    Note the "path" option on datasets is a bit inconsistent rn (mnist vs other). 

    get_dataloaders in setup.py (which calls data) also returns metadata - this is information which informs the construction of the model (like num channels). pass it into the model constructor, where each model will deal with it based on the specifications of the dataset file in datasets/dataset.py

    Hyperparameter scheduler and a loss function are also passed into the model constructed there. Models right now do not have an "automatic" registry, so you will need to edit setup.py manually. Furthermore, the hyperparameters for each model may be very different (in model config), so you need to deal with those individually for your model. 

    The loss function is customizable (to an extent): you have to specify the loss "type" to get a general type of loss function for a given task - right now you need to manually add these to the registry in losses.py. Loss functions take in hyperparameters as input, so you can use the scheduler in the model compute_loss to add different params. 

    The Optimizer - needs to be added to setup.py, don' thave custom functionality yet. 
    LR Scheduler - needs to be added to setup.py don't have custom functionality yet. 

    Then, all this is passed to the Trainer, which starts the training run. 
    This is where diagnostics are saved. 
    The logger contains ways of saving the plot and sany scalars you might want to keep. There's a whole fiasco about dynamic vs static naming. 



