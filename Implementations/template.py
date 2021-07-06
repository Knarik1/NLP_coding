import math

import torch
from torch import Tensor
import torch.nn as nn
import torchtext


if __name__ == '__main__':
    """
     Used sources
        https://pytorch.org/tutorials/beginner/translation_transformer.html
        
    """
    print(torch.__version__)  # 1.9.0+cu102
    print(torchtext.__version__)  # 0.10.0
