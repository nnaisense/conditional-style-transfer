import torch
import torch.distributions
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from . import layers
from scipy.ndimage.filters import gaussian_filter
import time

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance', dims = 2):
    if dims == 2:
        bnlayer = nn.BatchNorm2d
        inlayer = nn.InstanceNorm2d
    elif dims == 1:
        bnlayer = nn.BatchNorm1d
        inlayer = nn.InstanceNorm1d
    else:
        raise NotImplementedError('%i-D normalization layer is not found' % dims)
        
    if norm_type == 'batch':
        norm_layer = functools.partial(bnlayer, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(inlayer, affine=False, track_running_stats=False)
    elif norm_type == 'instance_affine':
        norm_layer = functools.partial(inlayer, affine=True, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # 1.0 - max(0, epoch + 1 - 100) / (101)
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], knn=5, alpha=0.5, peer_reg='basic', no_global_style=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netG == 'resnet_residual':
        net = ResnetGeneratorResidual(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, knn=knn, peer_reg=peer_reg, no_global_style=no_global_style, decoder='basic')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[], noise_mag=0.1):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'disc_basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'disc_noisy':
        net = NLayerMultiDiscriminatorNoisy(input_nc * 2, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, noise_mag=noise_mag)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Encoders
##############################################################################
def create_encoder_model_multiscale(n_downsampling, n_blocks, ngf, padding_type, norm_layer, use_dropout, use_bias, input_nc):    
    model = [nn.ReflectionPad2d(3),
             nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                       bias=use_bias),
             norm_layer(ngf),
             nn.ReLU()]

    i = 0
    mult = 2**i
    model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                        stride=2, padding=1, bias=use_bias),
              norm_layer(ngf * mult * 2),
              nn.ReLU()]
    model_enc_pre = nn.Sequential(*model)
        
    model = []
    i = 1
    mult = 2**i
    model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                        stride=2, padding=1, bias=use_bias),
              norm_layer(ngf * mult * 2),
              nn.ReLU()]

    mult = 2**n_downsampling
    for i in range(n_blocks):
        model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
    N_cont = ngf * mult 
    N_style = ngf * mult * 2
        
    model += [nn.Conv2d(ngf * mult, (N_cont + N_style), kernel_size=1, padding=0, stride=1, bias=use_bias),
              norm_layer(N_cont + N_style),
              nn.ReLU()]

    model_enc = nn.Sequential(*model)
        
    return model_enc, model_enc_pre, N_cont, N_style, mult
        
def create_encoder_global_model(num_feats, norm_layer, use_bias, global_pool):
    model = []
    model += [nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1, stride=2, bias=use_bias),
              norm_layer(num_feats),
              nn.ReLU()]
    model += [nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1, stride=2, bias=use_bias),
              norm_layer(num_feats),
              nn.ReLU()]
    model += [nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1, stride=2, bias=use_bias),
              norm_layer(num_feats),
              nn.ReLU()]
    if global_pool:
        model += [nn.AvgPool2d((8, 8), stride=(1, 1))]
    return nn.Sequential(*model)

##############################################################################
# Decoders
##############################################################################        
def create_merge_model(N_cont, N_style, n_downsampling, ngf, mult, norm_layer, use_bias):
    model = []
    mult = 2**n_downsampling
    model += [nn.Conv2d(N_cont + N_style, ngf * mult, kernel_size=1, padding=0, stride = 1, bias=use_bias),
                   norm_layer(ngf * mult),
                   nn.ReLU()]
    return nn.Sequential(*model)
        
        
def create_decoder_model(n_downsampling, n_blocks, ngf, mult, padding_type, norm_layer, use_dropout, use_bias, output_nc):
    model = []
        
    mult = 2**n_downsampling
    for i in range(n_blocks):
        model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=4, stride=2,
                                     padding=1, output_padding=0,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU()]
    model += [nn.ReflectionPad2d(3)]
    model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
    model += [nn.Tanh()]
        
    return nn.Sequential(*model)
    
    
def create_decoder_model_upsample(n_downsampling, n_blocks, ngf, mult, padding_type, norm_layer, use_dropout, use_bias, output_nc):
    model = []
        
    mult = 2**n_downsampling
    for i in range(n_blocks):
        model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        model += [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=5, stride=1,
                                     padding=2, 
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU()]
    model += [nn.ReflectionPad2d(3)]
    model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
    model += [nn.Tanh()]
        
    return nn.Sequential(*model)
        

##############################################################################
# Models
##############################################################################
   
class ResnetGeneratorResidual(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', knn=5, peer_reg='basic', no_global_style=False, decoder='basic'):
        assert(n_blocks >= 0)
        super(ResnetGeneratorResidual, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.knn = 5
        self.no_global_style = no_global_style
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # ORIGINAL MODEL encoder
        # Define the encoder model part
        n_downsampling = 2
        self.model_enc, self.model_enc_pre, self.N_cont, self.N_style, mult = create_encoder_model_multiscale(n_downsampling, n_blocks, ngf, padding_type, norm_layer, use_dropout, use_bias, input_nc)
        self.model_enc_global = create_encoder_global_model(self.N_style // 2, norm_layer, use_bias, False)
        
        #################################
        model = []
        if peer_reg == 'basic':
            model += [layers.PeerRegularizationLayerAtt(att_dim = self.N_cont, att_dropout_rate = 0.2, K = knn)]
        elif peer_reg == 'bidir':
            model += [layers.PeerRegularizationLayerAttBidir(att_dim = self.N_cont, att_dropout_rate = 0.2, K = knn)]
        else:
            raise NotImplementedError('Peer regularization [%s] is not recognized' % peer_reg)
        self.model_peer_reg = nn.Sequential(*model)
        
        # Model merging content and style
        self.model_merge = create_merge_model(self.N_cont, self.N_style, n_downsampling, ngf, mult, norm_layer, use_bias)
        self.model_merge_aux = create_merge_model(self.N_cont, self.N_style, n_downsampling, ngf, mult, norm_layer, use_bias)
        #################################
                
        # ORIGINAL MODEL decoder
        # Define the decoder model part
        if decoder == 'basic':
            create_decoder = create_decoder_model
        else:
            raise NotImplementedError('Decoder model [%s] is not recognized' % decoder)
            
        self.model_dec = create_decoder(n_downsampling, n_blocks, ngf, mult, padding_type, norm_layer, use_dropout, use_bias, output_nc)
        self.model_dec_aux = create_decoder(n_downsampling, n_blocks, ngf, mult, padding_type, norm_layer, use_dropout, use_bias, output_nc)
        
    def freeze_params(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.freeze_params(child)
            
    def unfreeze_params(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True
            self.unfreeze_params(child)
        
    def optimize_aux(self):
        self.unfreeze_params(self.model_enc_pre)
        self.unfreeze_params(self.model_enc)
        self.unfreeze_params(self.model_enc_global)
        self.unfreeze_params(self.model_merge_aux)
        self.unfreeze_params(self.model_dec_aux)
        self.freeze_params(self.model_peer_reg)
        self.freeze_params(self.model_merge)
        self.freeze_params(self.model_dec)
    
    def optimize_main(self):
        self.freeze_params(self.model_enc_pre)
        self.freeze_params(self.model_enc)
        self.freeze_params(self.model_enc_global)
        self.freeze_params(self.model_merge_aux)
        self.freeze_params(self.model_dec_aux)
        self.unfreeze_params(self.model_peer_reg)
        self.unfreeze_params(self.model_merge)
        self.unfreeze_params(self.model_dec)
        
    def optimize_all(self):
        self.unfreeze_params(self.model_enc_pre)
        self.unfreeze_params(self.model_enc)
        self.unfreeze_params(self.model_enc_global)
        self.unfreeze_params(self.model_merge_aux)
        self.unfreeze_params(self.model_dec_aux)
        self.unfreeze_params(self.model_peer_reg)
        self.unfreeze_params(self.model_merge)
        self.unfreeze_params(self.model_dec)
        
    def get_style_feats(self, input):
        # STYLE encoder is external
        # Prepare style
        z = self.model_enc(input)
        z_C, z_S = z[:, :self.N_cont, :, :], z[:, self.N_cont:, :, :]
    
        return z_S
    
    def encode_style_global_local(self, input):
        if self.no_global_style:
            return input
        
        N, C, W, H = input.shape
        style_glob, style_loc = torch.split(input, C // 2, dim = 1)
        # GLobal average pooling to obtain the global style vector
        style_glob = self.model_enc_global(style_glob)#.expand(N, C // 2, W, H)
        style_glob = torch.mean(torch.mean(style_glob, dim = 2, keepdim = True), dim = 3, keepdim = True)
        style_glob = style_glob.expand(N, C // 2, W, H)
        
        return torch.cat([style_glob, style_loc], dim = 1)
       
    def encode(self, input):
        z_inp_pre = self.model_enc_pre(input)
        z_inp = self.model_enc(z_inp_pre)
        z_inp_C, z_inp_S = z_inp[:, :self.N_cont, :, :], z_inp[:, self.N_cont:, :, :]
        z_inp_S = self.encode_style_global_local(z_inp_S)
        
        return {'cont': z_inp_C, 'middle': z_inp_pre, 'style': z_inp_S}
        
    def decode(self, z_input):
        z_inp_combined = self.model_merge(z_input)
        rec = self.model_dec(z_inp_combined)
        return rec
    
    def decode_aux(self, z_input):
        z_inp_combined = self.model_merge_aux(z_input)
        rec = self.model_dec_aux(z_inp_combined)
        return rec
    
    def reconstruct_image(self, input, zero_cont = False, zero_style = False):
        inp = input
        
        # Encode input
        z_inp = self.encode(inp)
        z_new = torch.cat([z_inp['cont'].detach(), z_inp['style'].detach()], dim = 1)
        
        if zero_cont:
            z_new[:, :self.N_cont, :, :] = torch.zeros_like(z_new[:, :self.N_cont, :, :])
            
        if zero_style:
            z_new[:, self.N_cont:, :, :] = torch.zeros_like(z_new[:, self.N_cont:, :, :])
        
        # Merge and decode
        rec = self.decode(z_new.cuda())
        
        return rec
    
    def stylize_image(self, input, cont_transf = True, style_transf = True):
        inp, tgt = input
        
        # Encode input
        z_inp = self.encode(inp)
        
        # Encode style
        z_tgt = self.encode(tgt)
        
        # Peer regularize
        z_new = self.model_peer_reg([[z_inp['cont'].detach(), z_inp['style'].detach()], [z_tgt['cont'].detach(), z_tgt['style'].detach()], cont_transf, style_transf])
        z_new_S = z_new[:, self.N_cont:, :, :]
        
        # Merge and decode
        rec = self.decode(z_new.cuda())
        
        return rec, z_inp['cont'], z_inp['style'], z_tgt['cont'], z_tgt['style']
        
        
    def forward(self, input):
        #cont, cond_id, style_id, style, style_other
        inp_src, inp_tgt, dst_src, dst_tgt, dst_other = input
        
        # Input image (1 x ch x w x h)
        # Style images (num_style_imgs x ch x w x h)
                
        # Dictionary to store all the features
        feats = {}
            
        # INPUT encoder
        # Encode source inpt
        feats['inp_src'] = self.encode(inp_src)
        
        # Encode target input
        feats['inp_tgt'] = self.encode(inp_tgt)
        
        # Encode source dest
        feats['dst_src'] = self.encode(dst_src)
        
        # Encode target dest
        feats['dst_tgt'] = self.encode(dst_tgt)
        
        # Peer regularize
        z_cont_new = self.model_peer_reg([[feats['inp_src']['cont'].detach(), feats['inp_src']['style'].detach()], [feats['dst_tgt']['cont'].detach(), feats['dst_tgt']['style'].detach()]])        
        feats['peer_reg'] = {'cont': z_cont_new[:, :self.N_cont, :, :], 'middle': None, 'style': z_cont_new[:, self.N_cont:, :, :]}
        
        # Merge and decode
        rec = self.decode(z_cont_new)
        
        # Perform style update for consistency loss
        feats['rec'] = self.encode(rec)
        
        # Identity mapping
        ##############
        # Peer regularize
        z_cont_new_id = torch.cat([feats['inp_src']['cont'].detach(), feats['inp_src']['style'].detach()], dim = 1)
        
        # Merge and decode identity
        rec_id = self.decode(z_cont_new_id)
        
        # Perform style update for consistency loss
        feats['rec_id'] = self.encode(rec_id)
        
        ##############
        # Peer regularize
        z_style_new_id = torch.cat([feats['dst_tgt']['cont'].detach(), feats['dst_tgt']['style'].detach()], dim = 1)
        
        # Merge and decode identity
        rec_style_id = self.decode(z_style_new_id)
        
        # Perform style update for consistency loss
        feats['rec_style_id'] = self.encode(rec_style_id)
        ##############
        
        # Encode style of other classes
        if dst_other is not None:
            feats['dst_other'] = self.encode(dst_other)
        else:
            feats['dst_other'] = None
        
        # Auxiliary decoder
        z_cont_new_id_aux = torch.cat([feats['inp_src']['cont'], feats['inp_src']['style']], dim = 1)
        rec_id_aux = self.decode_aux(z_cont_new_id_aux)
        feats['rec_id_aux'] = self.encode(rec_id_aux)
        
        # Auxiliary decoder processing style
        z_style_new_id_aux = torch.cat([feats['dst_tgt']['cont'], feats['dst_tgt']['style']], dim = 1)
        rec_style_id_aux = self.decode_aux(z_style_new_id_aux)
        feats['rec_style_id_aux'] = self.encode(rec_style_id_aux)
        
        return rec, rec_id, rec_style_id, rec_id_aux, rec_style_id_aux, feats
    
    
###########################################################################
        
# Code adapted from from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    
#########################################################################################################
#########################################################################################################
#########################################################################################################

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


    
class NLayerMultiDiscriminatorNoisy(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, noise_mag = 0.01):
        super(NLayerMultiDiscriminatorNoisy, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        self.noise_mag = noise_mag
                        
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model_pred = nn.Sequential(*sequence)

        
    def forward(self, input):
        fts, rfts = input
        
        if rfts is not None:
            inp = torch.cat([fts, rfts], dim = 1)
        else:
            inp = fts
        
        noise = torch.randn_like(inp) * self.noise_mag
        inp = inp + noise
        
        pred = self.model_pred(inp)
        
        return [pred]
    
    