import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import layers


class GANAuxModel(BaseModel):
    def name(self):
        return 'GANAuxModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default GANAux does not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_gen', type=float, default=1.0, help='weight for generator loss')
            parser.add_argument('--lambda_disc', type=float, default=1.0, help='weight for discriminator loss')
            parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_cont', type=float, default=1.0, help='weight for content latent loss')
            parser.add_argument('--lambda_idt', type=float, default=1.0, help='weight for identity loss')
            parser.add_argument('--lambda_idt_aux', type=float, default=1.0, help='weight for aux identity loss')
            parser.add_argument('--lambda_cycle_aux', type=float, default=1.0, help='weight for aux cycle loss')
            parser.add_argument('--lambda_transf', type=float, default=1.0, help='weight for transf loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['idt', 'disc', 'z_style', 'gen', 'z_cont', 'idt_aux', 'cycle_aux', 'transf']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'rec_A', 'fake_B', 'rec_A_aux']
        visual_names_B = ['style_B_0', 'style_B_1', 'style_B_2', 'style_A_0', 'style_A_1', 'style_A_2']
        
        self.visual_names = visual_names_A + visual_names_B 
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time
            self.model_names = ['G', 'D']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, knn=opt.knn, peer_reg=opt.peer_reg, no_global_style=opt.no_global_style)
        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids, noise_mag=opt.D_noise_mag)
            
        if self.isTrain:
            self.pool_A = ImagePool(opt.pool_size)
            self.pool_B = ImagePool(opt.pool_size)
            
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()
            self.criterionSmoothL1Elemwise = torch.nn.SmoothL1Loss(reduction = 'none')
            
            # initialize optimizers
            self.netG.module.optimize_aux()
            self.optimizer_G_aux = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.netG.parameters())),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)
            self.netG.module.optimize_main()
            self.optimizer_G_main = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.netG.parameters())),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)
            self.netG.module.optimize_all()
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.netD.parameters())),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_aux)
            self.optimizers.append(self.optimizer_G_main)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.style_A = input['A_style'].to(self.device)
        self.style_B = input['B_style'].to(self.device)
        self.dummy = torch.zeros((1, 3, 256, 256))
        self.style_B_0 = self.style_B[0:1]
        self.style_B_1 = self.dummy
        self.style_B_2 = self.dummy
        self.style_A_0 = self.style_A[0:1]
        self.style_A_1 = self.dummy
        self.style_A_2 = self.dummy
        self.image_paths = input['A_paths']
        
    def forward(self):
        self.fake_B, self.rec_A, self.rec_style_B, self.rec_A_aux, self.rec_style_B_aux, self.feats = self.netG([self.real_A, self.real_B, self.style_A, self.style_B, None])
        
    def disc_loss(self, real, fake, style, mult = 1.0):
        loss = 0.0
        for i in range(style.shape[0]):
            preds = self.netD([real, style[i:i+1]])
            preds_fake = self.netD([fake, style[i:i+1]])
            subloss = 0.0
            for i in range(len(preds)):
                pred = preds[i]
                pred_fake = preds_fake[i]
                subloss += (torch.mean((pred - torch.mean(pred_fake) - mult * torch.ones_like(pred)) ** 2) + torch.mean((pred_fake - torch.mean(pred) + mult * torch.ones_like(pred)) ** 2)) / 2.0
            loss += (subloss / len(preds))
        loss /= style.shape[0]
        return loss
        
    def backward_D(self, retain_graph = False):
        pooled_A = self.pool_A.query(torch.cat([self.real_A, self.rec_A, self.style_A], dim = 1))
        pooled_B = self.pool_B.query(torch.cat([self.real_B, self.fake_B, self.style_B], dim = 1))
        real_A, rec_A, style_A = torch.split(pooled_A, 3, dim = 1)
        real_B, fake_B, style_B = torch.split(pooled_B, 3, dim = 1)
        
        self.loss_style_rec_A = self.disc_loss(real_A, rec_A.detach(), style_A, mult = 1.0)
        self.loss_style_fake_B = self.disc_loss(real_B, fake_B.detach(), style_B, mult = 1.0)
        
        self.loss_disc = (self.loss_style_rec_A + self.loss_style_fake_B) * self.opt.lambda_disc
        self.loss_disc /= 2.0
        
        self.loss_disc.backward(retain_graph=retain_graph)
        
    def backward_G_main(self, retain_graph = False):
        # Style transfer loss
        self.loss_gen_A = self.disc_loss(self.real_A, self.rec_A, self.style_A, mult = -1.0)
        self.loss_gen_B = self.disc_loss(self.real_B, self.fake_B, self.style_B, mult = -1.0)
            
        self.loss_gen = (self.loss_gen_A + self.loss_gen_B) * self.opt.lambda_gen
        self.loss_gen /= 2.0
                
        # Transformed image loss
        self.loss_idt = self.criterionL2(self.rec_A, self.real_A.detach()) * self.opt.lambda_idt
        self.loss_idt += self.criterionL2(self.rec_style_B, self.style_B.detach()) * self.opt.lambda_idt
        self.loss_idt /= 2.0
        
        # Transfer loss
        self.loss_transf = 0.0
        self.loss_transf += torch.mean(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec']['cont'], self.feats['inp_src']['cont'].detach()), dim = 1)) * self.opt.lambda_transf
        self.loss_transf += torch.mean(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec']['style'], self.feats['dst_tgt']['style'].detach()), dim = 1)) * self.opt.lambda_transf
        self.loss_transf /= 2.0
        
        # Combined loss
        self.loss_G = self.loss_gen + self.loss_idt + self.loss_transf
        
        self.loss_G.backward(retain_graph=retain_graph)
        
    def backward_G_aux(self, retain_graph = False):
        # Content latent loss
        self.loss_z_cont = 0.0
        pos_dists = torch.mean(torch.flatten(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec']['cont'], self.feats['inp_src']['cont'].detach()), dim = 1), start_dim = 1), dim = 1)
        pos_dists = torch.cat([pos_dists, torch.mean(torch.flatten(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec_id']['cont'], self.feats['inp_src']['cont'].detach()), dim = 1), start_dim = 1), dim = 1)], dim = 0)
        self.loss_z_cont += pos_dists.mean() * self.opt.lambda_cont
        
        # Style latent loss
        self.loss_z_style = 0.0
        dst_tgt_sty = self.feats['dst_tgt']['style']
        inp_tgt_sty = self.feats['inp_tgt']['style']
        dst_src_sty = self.feats['dst_src']['style']
        inp_src_sty = self.feats['inp_src']['style']
        pos_dists = torch.mean(torch.flatten(torch.sum(self.criterionSmoothL1Elemwise(dst_tgt_sty, inp_tgt_sty.expand(dst_tgt_sty.shape).detach()), dim = 1), start_dim = 1), dim = 1)
        pos_dists = torch.cat([pos_dists, torch.mean(torch.flatten(torch.sum(self.criterionSmoothL1Elemwise(dst_src_sty, inp_src_sty.expand(dst_src_sty.shape).detach()), dim = 1), start_dim = 1), dim = 1)], dim = 0)
        neg_dists = torch.mean(torch.flatten(torch.sum(self.criterionSmoothL1Elemwise(dst_src_sty, inp_tgt_sty.expand(dst_src_sty.shape).detach()), dim = 1), start_dim = 1), dim = 1)
        neg_dists = torch.cat([neg_dists, torch.mean(torch.flatten(torch.sum(self.criterionSmoothL1Elemwise(dst_tgt_sty, inp_src_sty.expand(dst_tgt_sty.shape).detach()), dim = 1), start_dim = 1), dim = 1)], dim = 0)
        self.loss_z_style += (pos_dists.mean() + torch.max(torch.tensor(0.0).to(self.device), self.opt.ml_margin - neg_dists.mean())) * self.opt.lambda_style
 
        # Cycle aux
        self.loss_cycle_aux = 0.0
        self.loss_cycle_aux += torch.mean(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec_id_aux']['cont'], self.feats['inp_src']['cont'].detach()), dim = 1)) * self.opt.lambda_cycle_aux
        self.loss_cycle_aux += torch.mean(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec_id_aux']['style'], self.feats['inp_src']['style'].detach()), dim = 1)) * self.opt.lambda_cycle_aux
        self.loss_cycle_aux += torch.mean(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec_style_id_aux']['style'], self.feats['dst_tgt']['style'].detach()), dim = 1)) * self.opt.lambda_cycle_aux
        self.loss_cycle_aux += torch.mean(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec_id_aux']['middle'], self.feats['inp_src']['middle'].detach()), dim = 1)) * self.opt.lambda_cycle_aux
        self.loss_cycle_aux += torch.mean(torch.sum(self.criterionSmoothL1Elemwise(self.feats['rec_style_id_aux']['middle'], self.feats['dst_tgt']['middle'].detach()), dim = 1)) * self.opt.lambda_cycle_aux
        self.loss_cycle_aux /= 5.0

        # Identity loss
        self.loss_idt_aux = self.criterionL2(self.rec_A_aux, self.real_A.detach()) * self.opt.lambda_idt_aux
        self.loss_idt_aux += self.criterionL2(self.rec_style_B_aux, self.style_B.detach()) * self.opt.lambda_idt_aux
        self.loss_idt_aux /= 2.0
        
        # Combined loss
        self.loss_G_aux = self.loss_idt_aux + self.loss_z_cont + self.loss_z_style + self.loss_cycle_aux
        
        self.loss_G_aux.backward(retain_graph=retain_graph)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G AUX
        self.netG.module.optimize_aux()
        self.optimizer_G_aux.zero_grad()
        self.backward_G_aux(retain_graph = True)
        self.optimizer_G_aux.step()
        # G MAIN
        self.netG.module.optimize_main()
        self.optimizer_G_main.zero_grad()
        self.backward_G_main()
        self.optimizer_G_main.step()
        # D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.netG.module.optimize_all()
