import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_
from tqdm import tqdm
from os import listdir, path
import numpy as np
import os, subprocess
from glob import glob
import cv2

from talkingface.model.layers import Conv2d, Conv2dTranspose, nonorm_Conv2d
from talkingface.model.abstract_talkingface import AbstractTalkingFace
from talkingface.data.dataprocess.wav2lip_process import Wav2LipPreprocessForInference, Wav2LipAudio
from talkingface.utils import ensure_dir

class PC_AVS(nn.Module):
    def __init__(self):
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        out=self.linear(x)
        return out

