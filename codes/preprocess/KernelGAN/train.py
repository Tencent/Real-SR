import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner


def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data[iteration]
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    gan.finish()


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='../ntire20/DPEDiphone-tr-x', help='path to image input directory.')
    # prog.add_argument('--input-dir', '-i', type=str, default='test_images',
    #                   help='path to image input directory.')
    prog.add_argument('--max_kernel_num', type=int, default=250, help='max kernel number')
    prog.add_argument('--output-dir', '-o', type=str, default='./preprocess/KernelGAN/results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    img_paths = os.listdir(os.path.abspath(args.input_dir))
    img_paths = img_paths[:max(len(img_paths), args.max_kernel_num)]
    path_num = len(img_paths)
    for id, filename in enumerate(img_paths):
        print('Processing - {}/{}'.format(id + 1, path_num))
        conf = Config().parse(create_params(filename, args))
        train(conf)
    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir)]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    return params


if __name__ == '__main__':
    main()
