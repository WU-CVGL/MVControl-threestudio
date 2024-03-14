import os
import numpy as np
import torch
import random
from accelerate.logging import get_logger
from accelerate.state import PartialState
from accelerate.utils import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    convert_outputs_to_fp32,
    get_pretty_name,
    is_tpu_available,
    save,
)


logger = get_logger(__name__)


# Only used when accelerator.distributed_type == DistributedType.MULTI_GPU
def load_state(accelerator, input_dir, map_location='cpu', ):
    # Check if folder exists
    input_dir = os.path.expanduser(input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Tried to find {input_dir} but folder does not exist")
    logger.info(f"Loading states from {input_dir}")

    models = accelerator._models
    optimizers = accelerator._optimizers
    schedulers = accelerator._schedulers

    # Call model loading hooks that might have been registered with
    # accelerator.register_model_state_hook
    for hook in accelerator._load_model_state_pre_hook.values():
        hook(models, input_dir)
    
    load_accelerator_state(
        input_dir,
        models,
        optimizers,
        schedulers,
        accelerator.state.process_index,
        accelerator.scaler,
        map_location,
    )

# Modified based on accelerate.checkpointing.load_accelerator_state
def load_accelerator_state(
    input_dir,
    models,
    optimizers,
    schedulers,
    process_index,
    scaler=None,
    map_location=None,
    **load_model_func_kwargs,
):
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.

    Args:
        input_dir (`str` or `os.PathLike`):
            The name of the folder to load all relevant weights and states.
        models (`List[torch.nn.Module]`):
            A list of model instances
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional *GradScaler* instance to load
        map_location (`str`, *optional*):
            What device to load the optimizer state onto. Should be one of either "cpu" or "on_device".
        load_model_func_kwargs (`dict`, *optional*):
            Additional arguments that can be passed to the model's `load_state_dict` method.
    """
    if map_location not in [None, "cpu", "on_device"]:
        raise TypeError(
            "Unsupported optimizer map location passed, please choose one of `None`, `'cpu'`, or `'on_device'`"
        )
    if map_location is None:
        map_location = "cpu"
    elif map_location == "on_device":
        map_location = PartialState().device
    # Model states
    for i, model in enumerate(models):
        weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
        input_model_file = os.path.join(input_dir, weights_name)
        models[i].load_state_dict(torch.load(input_model_file, map_location=map_location), **load_model_func_kwargs)
    logger.info("All model weights loaded successfully")

    # Optimizer states
    for i, opt in enumerate(optimizers):
        optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
        input_optimizer_file = os.path.join(input_dir, optimizer_name)
        optimizer_state = torch.load(input_optimizer_file)
        optimizers[i].load_state_dict(optimizer_state, map_location=map_location)
    logger.info("All optimizer states loaded successfully")

    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
        input_scheduler_file = os.path.join(input_dir, scheduler_name)
        scheduler.load_state_dict(torch.load(input_scheduler_file))
    logger.info("All scheduler states loaded successfully")

    # GradScaler state
    if scaler is not None:
        input_scaler_file = os.path.join(input_dir, SCALER_NAME)
        scaler.load_state_dict(torch.load(input_scaler_file))
        logger.info("GradScaler state loaded successfully")

    # Random states
    try:
        states = torch.load(os.path.join(input_dir, f"{RNG_STATE_NAME}_{process_index}.pkl"))
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        
        logger.info("All random states loaded successfully")
    except Exception:
        logger.info("Could not load random states")


def prepare_model_multigpu(model, accelerator, device_placement=None):

    if device_placement is None:
        device_placement = accelerator.device_placement
    # accelerator._models.append(model)
    # We check only for models loaded with `accelerate`

    # Checks if any of the child module has the attribute `hf_device_map`.
    has_hf_device_map = False
    for m in model.modules():
        if hasattr(m, "hf_device_map"):
            has_hf_device_map = True
            break

    if getattr(model, "is_loaded_in_8bit", False) and getattr(model, "hf_device_map", False):
        model_devices = set(model.hf_device_map.values())
        if len(model_devices) > 1:
            raise ValueError(
                "You can't train a model that has been loaded in 8-bit precision on multiple devices."
            )

        current_device_index = list(model_devices)[0]
        if torch.device(current_device_index) != accelerator.device:
            # if on the first device (GPU 0) we don't care
            if (accelerator.device.index is not None) or (current_device_index != 0):
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                    "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}"
                )

        if "cpu" in model_devices or "disk" in model_devices:
            raise ValueError(
                "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
            )
    elif device_placement and not has_hf_device_map:
        model = model.to(accelerator.device)

    if any(p.requires_grad for p in model.parameters()):
        kwargs = accelerator.ddp_handler.to_kwargs() if accelerator.ddp_handler is not None else {}
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[accelerator.local_process_index], output_device=accelerator.local_process_index, **kwargs
        )

    model._original_forward = model.forward
    if accelerator.mixed_precision == "fp16":
        model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
    else:
        model.forward = torch.cuda.amp.autocast()(model.forward)
    model.forward = convert_outputs_to_fp32(model.forward)

    return model