import os
from options.test_options import TestOptions
from data import CreateDataLoader, CreateStyleDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import imageio

from collections import OrderedDict


from PIL import Image
import torchvision.transforms as transforms

############################################################################
def get_transform(loadSize = 512, fineSize = 512, pad = None):
    transform_list = []
    
    transform_list.append(transforms.Resize(loadSize, Image.BICUBIC))
    transform_list.append(transforms.CenterCrop(fineSize))
    if pad is not None:
        transform_list.append(transforms.Pad(pad, padding_mode='reflect'))
    
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_image(A_path, transf_fn):        
    A_img = Image.open(A_path).convert('RGB')
    A = transf_fn(A_img)
    return A
############################################################################


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.eval = True
    
    # Create model
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()
    
    # Load the data to be evaluated
    pad_size = 8 * (opt.loadSize // 256)
    transform_fn = get_transform(loadSize = opt.loadSize, fineSize = opt.fineSize, pad = pad_size)
    transform_fn_style = get_transform(loadSize = opt.loadSize, fineSize = opt.fineSize)

    # Load content images and style images from the specified folders
    files = os.listdir(opt.content_folder)
    imgs = {}
    for afile in files:
        img = get_image(os.path.join(opt.content_folder, afile), transform_fn)
        imgs[os.path.splitext(afile)[0]] = img
    
    files = os.listdir(opt.style_folder)
    styles = {}
    for afile in files:
        style = get_image(os.path.join(opt.style_folder, afile), transform_fn_style)
        styles[os.path.splitext(afile)[0]] = style
    
    # Run the model
    for cont in imgs: 
        for sty in styles:
            print('Processing pair %s_%s ...' % (cont, sty))
            real_A = imgs[cont].unsqueeze(0)
            style_B = styles[sty].unsqueeze(0)
            with torch.no_grad():
                fake_B, z_cont_real_A, z_style_real_A, z_cont_style_B, z_style_B = model.netG.module.stylize_image([real_A.cuda(), style_B.cuda()])

                if pad_size is not None:
                    real_A = real_A[:, :, pad_size:-pad_size, pad_size:-pad_size]
                    fake_B = fake_B[:, :, pad_size:-pad_size, pad_size:-pad_size]

            out_dict = {
                'real_A': real_A.data.cpu().numpy()[0].transpose((1,2,0)), 'fake_B': fake_B.data.cpu().numpy()[0].transpose((1,2,0)), 
                'z_cont_real_A': z_cont_real_A.data.cpu().numpy(), 'z_cont_style_B': z_cont_style_B.data.cpu().numpy(),
                'z_style_real_A': z_style_real_A.data.cpu().numpy(), 'z_style_B': z_style_B.data.cpu().numpy(), 
                'style_B': style_B.data.cpu().numpy().transpose(0, 2, 3, 1)
            }
            
            imageio.imwrite(os.path.join(opt.output_folder, '%s_%s.png' % (cont, sty)), out_dict['fake_B'])
    
    print('Evaluation done.')
   