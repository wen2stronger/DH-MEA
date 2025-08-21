import logging
import torch
from datetime import datetime




def set_device(args):
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if device == 'cpu':
        logging.info("\n no gpu found, program is running on cpu! \n")
    return device