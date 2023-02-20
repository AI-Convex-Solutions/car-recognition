import torch
import gc


def clear_memory(instance):
    # We delete the following instances to free up memory for computation.
    del instance
    gc.collect()
    torch.cuda.empty_cache()
