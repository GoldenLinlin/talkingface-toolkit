from typing import Iterator
import torch
from torch.nn.parameter import Parameter
from talkingface.model.text_to_speech.pc_avs.base_network import PC_AVS
from talkingface.model.text_to_speech.pc_avs.loss import *
from talkingface.model.text_to_speech.pc_avs.discriminator import MultiscaleDiscriminator, ImageDiscriminator
from talkingface.model.text_to_speech.pc_avs.generator import ModulateGenerator
from talkingface.model.text_to_speech.pc_avs.encoder import ResSEAudioEncoder, ResNeXtEncoder, ResSESyncEncoder, FanEncoder
import talkingface.model.text_to_speech.pc_avs.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)
    netA_cls = find_network_using_name(opt.netA, 'encoder')
    parser = netA_cls.modify_commandline_options(parser, is_train)
    # parser = netA_sync_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_networks(opt, name, type):
    netG_cls = find_network_using_name(name, type)
    return create_network(netG_cls, opt)

def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)

def define_A(opt):
    netA_cls = find_network_using_name(opt.netA, 'encoder')
    return create_network(netA_cls, opt)

def define_A_sync(opt):
    netA_cls = find_network_using_name(opt.netA_sync, 'encoder')
    return create_network(netA_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name(opt.netE, 'encoder')
    return create_network(netE_cls, opt)


def define_V(opt):
    # there exists only one encoder type
    netV_cls = find_network_using_name(opt.netV, 'encoder')
    return create_network(netV_cls, opt)


def define_P(opt):
    netP_cls = find_network_using_name(opt.netP, 'encoder')
    return create_network(netP_cls, opt)


def define_F_rec(opt):
    netF_rec_cls = find_network_using_name(opt.netF_rec, 'encoder')
    return create_network(netF_rec_cls, opt)
import torch.nn as nn
from torch.nn import init


class PC_AVS(nn.Module):
    def __init__(self,a):
        super(PC_AVS, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

