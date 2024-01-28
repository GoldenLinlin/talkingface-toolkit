from typing import Iterator
import torch
import torch.nn
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
from options.test_options import TestOptions
import torch
from talkingface.model.text_to_speech.pc_avs import create_model
import data
import util.util as util
from tqdm import tqdm
import os
import sys
def video_concat(processed_file_savepath, name, video_names, audio_path):
    cmd = ['ffmpeg']
    num_inputs = len(video_names)
    for video_name in video_names:
        cmd += ['-i', '\'' + str(os.path.join(processed_file_savepath, video_name + '.mp4'))+'\'',]

    cmd += ['-filter_complex hstack=inputs=' + str(num_inputs),
            '\'' + str(os.path.join(processed_file_savepath, name+'.mp4')) + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)

    video_add_audio(name, audio_path, processed_file_savepath)


def video_add_audio(name, audio_path, processed_file_savepath):
    os.system('copy %* {} {}'.format(audio_path, processed_file_savepath))
    cmd = ['ffmpeg', '-i', '\'' + os.path.join(processed_file_savepath, name + '.mp4') + '\'',
                     '-i', audio_path,
                     '-q:v 0',
                     '-strict -2',
                     '\'' + os.path.join(processed_file_savepath, 'av' + name + '.mp4') + '\'',
                     '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)


def img2video(dst_path, prefix, video_path):
    cmd = ['ffmpeg', '-i', '\'' + video_path + '/' + prefix + '%d.jpg'
           + '\'', '-q:v 0', '\'' + dst_path + '/' + prefix + '.mp4' + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)


def inference_single_audio(opt, path_label, model):
    #
    opt.path_label = path_label
    dataloader = data.create_dataloader(opt)
    processed_file_savepath = dataloader.dataset.get_processed_file_savepath()

    idx = 0
    if opt.driving_pose:
        video_names = ['Input_', 'G_Pose_Driven_', 'Pose_Source_', 'Mouth_Source_']
    else:
        video_names = ['Input_', 'G_Fix_Pose_', 'Mouth_Source_']
    is_mouth_frame = os.path.isdir(dataloader.dataset.mouth_frame_path)
    if not is_mouth_frame:
        video_names.pop()
    save_paths = []
    for name in video_names:
        save_path = os.path.join(processed_file_savepath, name)
        util.mkdir(save_path)
        save_paths.append(save_path)
    for data_i in tqdm(dataloader):
        # print('==============', i, '===============')
        fake_image_original_pose_a, fake_image_driven_pose_a = model.forward(data_i, mode='inference')

        for num in range(len(fake_image_driven_pose_a)):
            util.save_torch_img(data_i['input'][num], os.path.join(save_paths[0], video_names[0] + str(idx) + '.jpg'))
            if opt.driving_pose:
                util.save_torch_img(fake_image_driven_pose_a[num],
                         os.path.join(save_paths[1], video_names[1] + str(idx) + '.jpg'))
                util.save_torch_img(data_i['driving_pose_frames'][num],
                         os.path.join(save_paths[2], video_names[2] + str(idx) + '.jpg'))
            else:
                util.save_torch_img(fake_image_original_pose_a[num],
                                    os.path.join(save_paths[1], video_names[1] + str(idx) + '.jpg'))
            if is_mouth_frame:
                util.save_torch_img(data_i['target'][num], os.path.join(save_paths[-1], video_names[-1] + str(idx) + '.jpg'))
            idx += 1

    if opt.gen_video:
        for i, video_name in enumerate(video_names):
            img2video(processed_file_savepath, video_name, save_paths[i])
        video_concat(processed_file_savepath, 'concat', video_names, dataloader.dataset.audio_path)

    print('results saved...' + processed_file_savepath)
    del dataloader
    return
class PC_AVS(nn.Module):
    def __init__(self, config):
        super(PC_AVS, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self,x):
        out=self.linear(x)
        return out
    def generate_batch():
        print("eeeeeeeeeeeeeeeeeeee")

    def parameters(self):
    # 获取模型中所有可学习的参数
        for param in self.children():
            if hasattr(param, 'parameters'):
                for p in param.parameters():
                    print("%%%%%%%%%%%%%%%%")
                    print(p)
                    yield p
    def generate_batch(self):

        print("fuck you")

