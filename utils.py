import gc
import logging
import traceback

import torch


def clear_memory(instance):
    # We delete the following instances to free up memory for computation.
    del instance
    gc.collect()
    torch.cuda.empty_cache()


def exc_handler(exctype, value, tb):
    logging.getLogger("").exception(
        ''.join(traceback.format_exception(exctype, value, tb))
    )
