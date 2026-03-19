import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)
import torch
import os

def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        logger.info(f'rank{torch.distributed.get_rank()}: ' + message, stacklevel=2)
    else:
        logger.info(message, stacklevel=2)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message, stacklevel=2)
    else:
        logger.info(message, stacklevel=2)


def find_latest_checkpoint(output_dir):
    """ Scan the output directory and return the latest checkpoint path """
    if not os.path.exists(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    if not checkpoints:
        return None

    # Sort by checkpoint number and return the latest one
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest_checkpoint


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch
