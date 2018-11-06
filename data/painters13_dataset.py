import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms

categories_names = \
        ['/a/arch', '/a/amphitheater', '/a/aqueduct', '/a/arena/rodeo', '/a/athletic_field/outdoor',
         '/b/badlands', '/b/balcony/exterior', '/b/bamboo_forest', '/b/barn', '/b/barndoor', '/b/baseball_field',
         '/b/beach', '/b/beach_house', '/b/beer_garden', '/b/boardwalk', '/b/boathouse',
         '/b/botanical_garden', '/b/bullring', '/b/butte', '/c/cabin/outdoor', '/c/campsite', '/c/campus',
         '/c/canal/natural', '/c/canal/urban', '/c/canyon', '/c/castle', '/c/church/outdoor', '/c/chalet',
         '/c/cliff', '/c/coast', '/c/corn_field', '/c/corral', '/c/cottage', '/c/courtyard', '/c/crevasse',
         '/d/dam', '/d/desert/vegetation', '/d/desert_road', '/d/doorway/outdoor', '/f/farm', 
         '/f/field/cultivated', '/f/field/wild', '/f/field_road', '/f/fishpond', '/f/florist_shop/indoor',
         '/f/forest/broadleaf', '/f/forest_path', '/f/forest_road', '/f/formal_garden', '/g/gazebo/exterior',
         '/g/glacier', '/g/golf_course', '/g/greenhouse/indoor', '/g/greenhouse/outdoor', '/g/grotto',
         '/h/hayfield', '/h/hot_spring', '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_floe',
         '/i/ice_shelf', '/i/iceberg', '/i/inn/outdoor', '/i/islet', '/j/japanese_garden', '/k/kasbah',
         '/k/kennel/outdoor', '/l/lagoon', '/l/lake/natural', '/l/lawn', '/l/library/outdoor', '/l/lighthouse',
         '/m/mansion', '/m/marsh', '/m/mausoleum', '/m/moat/water', '/m/mosque/outdoor', '/m/mountain',
         '/m/mountain_path', '/m/mountain_snowy', '/o/oast_house', '/o/ocean', '/o/orchard', '/p/park',
         '/p/pasture', '/p/pavilion', '/p/picnic_area', '/p/pier', '/p/pond', '/r/raft', '/r/railroad_track',
         '/r/rainforest', '/r/rice_paddy', '/r/river', '/r/rock_arch', '/r/roof_garden', '/r/rope_bridge',
         '/r/ruin', '/s/schoolhouse', '/s/sky', '/s/snowfield', '/s/swamp', '/s/swimming_hole',
         '/s/synagogue/outdoor', '/t/temple/asia', '/t/topiary_garden', '/t/tree_farm', '/t/tree_house',
         '/u/underwater/ocean_deep', '/u/utility_room', '/v/valley', '/v/vegetable_garden', '/v/viaduct',
         '/v/village', '/v/vineyard', '/v/volcano', '/w/waterfall', '/w/watering_hole', '/w/wave',
         '/w/wheat_field', '/z/zen_garden', '/a/alcove', '/a/artists_loft',
         '/b/building_facade', '/c/cemetery']

class Painters13Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        self.transform = get_transform(opt)
        
        # Get the real photo paths
        if opt.phase == 'train':
            photo_paths = []
            folder_path = os.path.join(opt.dataroot, 'train0')
            for cat in categories_names:
                cat_path = os.path.join(folder_path, cat[1:])

                imgpths = os.listdir(cat_path)
                for imgpth in imgpths:
                    photo_paths.append(os.path.join(cat_path, imgpth))
        else:
            photo_paths = make_dataset(os.path.join(opt.dataroot, opt.phase + '0'))
            photo_paths = sorted(photo_paths)
        
        self.dirs = [os.path.join(opt.dataroot, opt.phase + '0'),
                    os.path.join(opt.dataroot, opt.phase + '1'),
                    os.path.join(opt.dataroot, opt.phase + '2'),
                    os.path.join(opt.dataroot, opt.phase + '3'),
                    os.path.join(opt.dataroot, opt.phase + '4'),
                    os.path.join(opt.dataroot, opt.phase + '5'),
                    os.path.join(opt.dataroot, opt.phase + '6'),
                    os.path.join(opt.dataroot, opt.phase + '7'),
                    os.path.join(opt.dataroot, opt.phase + '8'),
                    os.path.join(opt.dataroot, opt.phase + '9'),
                    os.path.join(opt.dataroot, opt.phase + '10'),
                    os.path.join(opt.dataroot, opt.phase + '11'),
                    os.path.join(opt.dataroot, opt.phase + '12'),
                    os.path.join(opt.dataroot, opt.phase + '13')]
        
        self.paths = []
        self.paths.append(photo_paths)
        for adir in self.dirs[1:]:
            paths = make_dataset(adir)
            paths = sorted(paths)
            self.paths.append(paths)
            
        self.sizes = []
        for apath in self.paths:
            self.sizes.append(len(apath))
            
        self.idxs = np.random.permutation(range(1, len(self.dirs)))
        self.cntr = 0
         
    def custom_transform(self):
        transform_list = []
        
        if self.opt.isTrain: 
            scale = np.random.uniform(low=0.9, high=1.1)
            loadSize = int(self.opt.loadSize * scale)
        else:
            loadSize = self.opt.loadSize
        
        transform_list.append(transforms.Resize(loadSize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(self.opt.fineSize))
        
        if self.opt.isTrain and not self.opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
            
        if self.opt.isTrain: 
            transform_list += [transforms.ColorJitter(brightness=0.05, 
                                                      contrast=0.05, 
                                                      saturation=0.05, 
                                                      hue=0.05)]

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
    
    def get_images(self, A_path, B_path):        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        transf_fn = self.custom_transform()
        A = transf_fn(A_img)
        B = transf_fn(B_img)
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
            
        return A, B
    
    def __getitem__(self, index):
        if self.opt.isTrain: 
            np.random.seed(index)
        if self.opt.mapping_mode == 'one_to_all':
            pick_idx = index % (len(self.idxs) - 1)
            curr_idxs = self.idxs[pick_idx:pick_idx+2]
            self.cntr += 1
            if self.cntr >= len(self.idxs):
                self.cntr = 0
                self.idxs = np.random.permutation(range(1, len(self.dirs)))

            classA = 0
            classB = curr_idxs[0]
            self.A_paths = self.paths[0]
            self.B_paths = self.paths[curr_idxs[0]]
            self.A_size = self.sizes[0]
            self.B_size = self.sizes[curr_idxs[0]]
            self.C_idxs = curr_idxs[1:2]
        else:
            idxs = np.random.permutation(range(len(self.dirs)))
            classA = idxs[0]
            classB = idxs[1]
            self.A_paths = self.paths[idxs[0]]
            self.B_paths = self.paths[idxs[1]]
            self.A_size = self.sizes[idxs[0]]
            self.B_size = self.sizes[idxs[1]]
            self.C_idxs = idxs[2:3]
        
        # Content images
        index_A = random.randint(0, self.A_size - 1)
        index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        A, B = self.get_images(A_path, B_path)
        
        # Style images
        idxsA = np.array(np.random.permutation(range(len(self.A_paths))))
        idxsB = np.array(np.random.permutation(range(len(self.B_paths))))
        A_style_paths = [self.A_paths[i] for i in idxsA[0:self.opt.num_style_samples]]
        B_style_paths = [self.B_paths[i] for i in idxsB[0:self.opt.num_style_samples]] 
        
        A_style_imgs, B_style_imgs = [], []
        for i in range(len(A_style_paths)):
            A_style, B_style = self.get_images(A_style_paths[i], B_style_paths[i])
            A_style_imgs.append(A_style[np.newaxis, :, :, :])
            B_style_imgs.append(B_style[np.newaxis, :, :, :])
            
        A_style_imgs = np.concatenate(A_style_imgs, axis = 0)
        B_style_imgs = np.concatenate(B_style_imgs, axis = 0)
        
        # Additional style images from other classes of painters 
        C_style_imgs = []
        #cidxs = np.random.permutation(range(len(self.C_idxs)))
        #for i in range(self.opt.num_style_samples):
        for cidx in self.C_idxs:
            #cidx = cidxs[i]
            idxsC = np.array(np.random.permutation(range(len(self.paths[cidx]))))
            C_style_paths = [self.paths[cidx][i] for i in idxsC[0:1]]
            for i in range(len(C_style_paths)):
                C_style, _ = self.get_images(C_style_paths[i], C_style_paths[i])
                C_style_imgs.append(C_style[np.newaxis, :, :, :])
        C_style_imgs = np.concatenate(C_style_imgs, axis = 0)  
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_style': A_style_imgs, 'B_style': B_style_imgs, 'C_style': C_style_imgs, 'A_class': classA, 'B_class': classB,
               'A_style_paths': A_style_paths, 'B_style_paths': B_style_paths}

    def __len__(self):
        return 6144#max(self.sizes)

    def name(self):
        return 'Painters13Dataset'
