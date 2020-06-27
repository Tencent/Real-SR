import argparse
import torch
import os


# noinspection PyPep8
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # Paths
        self.parser.add_argument('--img_name', default='image1', help='image name for saving purposes')
        self.parser.add_argument('--input_image_path', default=os.path.dirname(__file__) + '/training_data/input.png', help='path to one specific image file')
        self.parser.add_argument('--output_dir_path', default=os.path.dirname(__file__) + '/results', help='results path')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=64, help='Generators crop size')
        self.parser.add_argument('--scale_factor', type=float, default=0.5, help='The downscaling scale factor')
        self.parser.add_argument('--X4', action='store_true', help='The wanted SR scale factor')

        # Network architecture
        self.parser.add_argument('--G_chan', type=int, default=64, help='# of channels in hidden layer in the G')
        self.parser.add_argument('--D_chan', type=int, default=64, help='# of channels in hidden layer in the D')
        self.parser.add_argument('--G_kernel_size', type=int, default=13, help='The kernel size G is estimating')
        self.parser.add_argument('--D_n_layers', type=int, default=7, help='Discriminators depth')
        self.parser.add_argument('--D_kernel_size', type=int, default=7, help='Discriminators convolution kernels size')

        # Iterations
        self.parser.add_argument('--max_iters', type=int, default=3, help='# of iterations')

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=2e-4, help='initial learning rate for discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')

        # GPU
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

        # Kernel post processing
        self.parser.add_argument('--n_filtering', type=float, default=40, help='Filtering small values of the kernel')

        # ZSSR configuration
        self.parser.add_argument('--do_ZSSR', action='store_true', help='when activated - ZSSR is not performed')
        self.parser.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
        self.parser.add_argument('--real_image', action='store_true', help='ZSSRs configuration is for real images')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        self.clean_file_name()
        self.set_output_directory()
        self.conf.G_structure = [7, 5, 3, 1, 1, 1]
        print("Scale Factor: %s \tZSSR: %s \tReal Image: %s" % (('X4' if self.conf.X4 else 'X2'), str(self.conf.do_ZSSR), str(self.conf.real_image)))
        return self.conf

    def clean_file_name(self):
        """Retrieves the clean image file_name for saving purposes"""
        self.conf.img_name = self.conf.input_image_path.split('/')[-1].replace('ZSSR', '') \
            .replace('real', '').replace('__', '').split('_.')[0].split('.')[0]

    def set_gpu_device(self):
        """Sets the GPU device if one is given"""
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)

    def set_output_directory(self):
        """Define the output directory name and create the folder"""
        self.conf.output_dir_path = os.path.join(self.conf.output_dir_path, self.conf.img_name)
        # In case the folder exists - stack 'l's to the folder name
        while os.path.isdir(self.conf.output_dir_path):
            self.conf.output_dir_path += 'l'
        os.makedirs(self.conf.output_dir_path)
