import torch
import json
import os
import pytorch_lightning as pl

from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name)
        logger.watch(training_wrapper)
    
        if args.save_dir and isinstance(logger.experiment.name, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.name, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project_name=args.name)
        checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    # Checkpoint callback configuration based on validation availability
    if val_dl:
        print("Using validation checkpoint callback")
        ckpt_params = {
            "dirpath": checkpoint_dir,
            "save_top_k": args.save_top_k,
            "monitor": 'val/avg_loss',
            "mode": 'min',
            "filename": '{epoch}-{step}'
        }
        
        # Check which checkpoint frequency to use
        if args.checkpoint_every_n_epoch > 0:
            ckpt_params["every_n_epochs"] = args.checkpoint_every_n_epoch
        elif args.checkpoint_every > 0:
            ckpt_params["every_n_train_steps"] = args.checkpoint_every
        else:
            # Default to check every epoch if neither is specified
            ckpt_params["every_n_epochs"] = 1
            
        ckpt_callback = pl.callbacks.ModelCheckpoint(**ckpt_params)
    else:
        print("Using training checkpoint callback")
        # For non-validation case, use the same logic
        if args.checkpoint_every_n_epoch > 0:
            ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=args.checkpoint_every_n_epoch, dirpath=checkpoint_dir, save_top_k=-1)
        else:
            ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    
    # Early stopping configuration if validation is available
    callbacks = [ckpt_callback, exc_callback, save_model_config_callback]
    
    if val_dl and args.early_stopping:
        print("Using early stopping callback")
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='train/loss',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)

    if args.val_dataset_config:
        print("Using validation demo callback")
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        print("Using training demo callback")
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)
    
    # Add demo_callback to the callbacks list
    callbacks.append(demo_callback)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    # Manually load checkpoint if path is provided, bypassing Lightning's loader
    if args.ckpt_path:
        print(f"Manually loading checkpoint from: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        # Check if state_dict key exists, common in Lightning checkpoints
        state_dict = checkpoint.get('state_dict', checkpoint) 
        # You might need error handling or key adjustment here if loading fails
        try:
            training_wrapper.load_state_dict(state_dict)
            print("Successfully loaded state_dict into training_wrapper.")
            # Set ckpt_path to None as we've already loaded
            resume_path = None 
        except RuntimeError as e:
            print(f"Error loading state_dict directly: {e}")
            print("Attempting to load with strict=False")
            # Try loading non-strictly if direct loading fails (e.g., missing/extra keys)
            training_wrapper.load_state_dict(state_dict, strict=False)
            print("Loaded state_dict into training_wrapper with strict=False.")
            # Set ckpt_path to None as we've already loaded
            resume_path = None 
    else:
        resume_path = None # No checkpoint to resume from

    val_args = {}
    
    # Manage validation by number of epochs
    if args.val_every_n_epoch > 0:
        val_args.update({
            "check_val_every_n_epoch": args.val_every_n_epoch,
            "val_check_interval": None,
        })
    # Manage validation by number of steps
    elif args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=callbacks,  # Using the callbacks list defined above
        logger=logger,
        log_every_n_steps=1,
        max_epochs=model_config["training"]["max_epochs"],
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        num_sanity_val_steps=2 if val_dl else 0,  # Testing validation before starting if available
        **val_args      
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=resume_path)

if __name__ == '__main__':
    main()
