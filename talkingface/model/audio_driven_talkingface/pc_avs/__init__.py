import torch
import os
import talkingface.model.audio_driven_talkingface.pc_avs.models.networks as networks

from talkingface.model.audio_driven_talkingface.pc_avs.models.networks.architecture import VGGFace19
import talkingface.model.audio_driven_talkingface.pc_avs.util.util as util
from talkingface.model.audio_driven_talkingface.pc_avs.models.networks.loss import CrossEntropyLoss
import sys
import argparse
import math
import os
import torch
import torch.nn as nn
import talkingface.model.audio_driven_talkingface.pc_avs.models as models
import talkingface.model.audio_driven_talkingface.pc_avs.data as data 
import pickle
class PC_AVS(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(PC_AVS, self).__init__()
        self.linear = nn.Linear(10, 5)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    # |data|: dictionary of the input data
    def parameters(self):
    # 获取模型中所有可学习的参数
        for param in self.children():
            if hasattr(param, 'parameters'):
                for p in param.parameters():
                    print("%%%%%%%%%%%%%%%%")
                    print(p)
                    yield p

        