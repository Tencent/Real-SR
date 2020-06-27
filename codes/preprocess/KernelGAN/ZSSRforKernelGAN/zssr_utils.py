import numpy as np
from math import pi, sin, cos
from cv2 import warpPerspective, INTER_CUBIC
from imresize import imresize
from scipy.ndimage import measurements, interpolation
from scipy.io import loadmat
from scipy.signal import convolve2d
import tensorflow as tf
from random import sample


def random_augment(ims,
                   base_scales=None,
                   leave_as_is_probability=0.2,
                   no_interpolate_probability=0.3,
                   min_scale=0.5,
                   max_scale=1.0,
                   allow_rotation=True,
                   scale_diff_sigma=0.01,
                   shear_sigma=0.01,
                   crop_size=128,
                   allow_scale_in_no_interp=False,
                   crop_center=None,
                   loss_map_sources=None):
    # Determine which kind of augmentation takes place according to probabilities
    random_chooser = np.random.rand()

    # Option 1: No augmentation, return the original image
    if random_chooser < leave_as_is_probability:
        mode = 'leave_as_is'

    # Option 2: Only non-interpolated augmentation, which means 8 possible augmentations (4 rotations X 2 mirror flips)
    elif leave_as_is_probability < random_chooser < leave_as_is_probability + no_interpolate_probability:
        mode = 'no_interp'

    # Option 3: Affine transformation (uses interpolation)
    else:
        mode = 'affine'

    # If scales not given, calculate them according to sizes of images. This would be suboptimal, because when scales
    # are not integers, different scales can have the same image shape.
    if base_scales is None:
        base_scales = [np.sqrt(np.prod(im.shape) / np.prod(ims[0].shape)) for im in ims]

    # In case scale is a list of scales with take the smallest one to be the allowed minimum
    max_scale = np.min([max_scale])

    # Determine a random scale by probability
    if mode == 'leave_as_is':
        scale = 1.0
    else:
        scale = np.random.rand() * (max_scale - min_scale) + min_scale

    # The image we will use is the smallest one that is bigger than the wanted scale
    # (Using a small value overlap instead of >= to prevent float issues)
    scale_ind, base_scale = next((ind, np.min([base_scale])) for ind, base_scale in enumerate(base_scales)
                                 if np.min([base_scale]) > scale - 1.0e-6)
    im = ims[scale_ind]

    # Next are matrices whose multiplication will be the transformation. All are 3x3 matrices.

    # First matrix shifts image to center so that crop is in the center of the image
    shift_to_center_mat = np.array([[1, 0, - im.shape[1] / 2.0],
                                    [0, 1, - im.shape[0] / 2.0],
                                    [0, 0, 1]])

    shift_back_from_center = np.array([[1, 0, im.shape[1] / 2.0],
                                       [0, 1, im.shape[0] / 2.0],
                                       [0, 0, 1]])
    # Keeping the transform interpolation free means only shifting by integers
    if mode != 'affine':
        shift_to_center_mat = np.round(shift_to_center_mat)
        shift_back_from_center = np.round(shift_back_from_center)

    # Scale matrix. if affine, first determine global scale by probability, then determine difference between x scale
    # and y scale by gaussian probability.
    if mode == 'affine' or (mode == 'no_interp' and allow_scale_in_no_interp):
        scale /= base_scale
        scale_diff = np.random.randn() * scale_diff_sigma
    else:
        scale = 1.0
        scale_diff = 0.0

    # In this matrix we also incorporate the possibility of mirror reflection (unless leave_as_is).
    if mode == 'leave_as_is' or not allow_rotation:
        reflect = 1
    else:
        reflect = np.sign(np.random.randn())

    scale_mat = np.array([[reflect * (scale + scale_diff / 2), 0, 0],
                          [0, scale - scale_diff / 2, 0],
                          [0, 0, 1]])
    # If center of crop was provided
    if crop_center is not None:
        shift_y, shift_x = crop_center
        shift_x = shift_x - crop_size / 2
        shift_y = shift_y - crop_size / 2
    # Shift matrix, this actually creates the random crop
    else:
        shift_x = np.random.rand() * np.clip(scale * im.shape[1] - crop_size, 0, 9999)
        shift_y = np.random.rand() * np.clip(scale * im.shape[0] - crop_size, 0, 9999)

    # Rotation matrix angle. if affine, set a random angle. if no_interp then theta can only be pi/2 times int.
    rotation_indicator = 0  # used for finding the correct crop
    if mode == 'affine':
        theta = np.random.rand() * 2 * pi
    elif mode == 'no_interp':
        rotation_indicator = np.random.randint(4)
        theta = rotation_indicator * pi / 2
    else:
        theta = 0

    if not allow_rotation:
        theta = 0

    # Rotation matrix structure
    rotation_mat = np.array([[cos(theta), sin(theta), 0],
                             [-sin(theta), cos(theta), 0],
                             [0, 0, 1]])

    if crop_center is not None:
        tmp_shift_y = shift_y
        rotation_indicator = (rotation_indicator * reflect) % 4
        if rotation_indicator == 1:
            shift_y = im.shape[1] - shift_x - crop_size
            shift_x = tmp_shift_y
        elif rotation_indicator == 2:
            shift_y = im.shape[0] - shift_y - crop_size
            shift_x = im.shape[1] - shift_x - crop_size
        elif rotation_indicator == 3:
            shift_y = shift_x
            shift_x = im.shape[0] - tmp_shift_y - crop_size

    shift_mat = np.array([[1, 0, - shift_x],
                          [0, 1, - shift_y],
                          [0, 0, 1]])
    # Keeping the transform interpolation free means only shifting by integers
    if mode != 'affine':
        shift_mat = np.round(shift_mat)

    # Shear Matrix, only for affine transformation.
    if mode == 'affine' and allow_rotation:
        shear_x = np.random.randn() * shear_sigma
        shear_y = np.random.randn() * shear_sigma
    else:
        shear_x = shear_y = 0
    shear_mat = np.array([[1, shear_x, 0],
                          [shear_y, 1, 0],
                          [0, 0, 1]])

    # Create the final transformation by multiplying all the transformations.
    transform_mat = (shift_back_from_center
                     .dot(shift_mat)
                     .dot(shear_mat)
                     .dot(rotation_mat)
                     .dot(scale_mat)
                     .dot(shift_to_center_mat))

    # Apply transformation to image and return the transformed image clipped between 0-1
    return np.clip(warpPerspective(im, transform_mat, (crop_size, crop_size), flags=INTER_CUBIC), 0, 1), \
           warpPerspective(loss_map_sources[scale_ind], transform_mat, (crop_size, crop_size), flags=INTER_CUBIC)


def preprocess_kernels(kernels, conf):
    # Load kernels if given files. if not just use the downscaling method from the configs.
    # output is a list of kernel-arrays or a a list of strings indicating downscaling method.
    # In case of arrays, we shift the kernels (see next function for explanation why).
    # Kernel is a .mat file (MATLAB) containing a variable called 'Kernel' which is a 2-dim matrix.
    if kernels is not None:
        return [kernel_shift(loadmat(kernel)['Kernel'], sf)
                for kernel, sf in zip(kernels, conf.scale_factors)]
    else:
        return [conf.downscale_method] * len(conf.scale_factors)


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def tensorshave(im, margin):
    shp = tf.shape(im)
    if shp[3] == 3:
        return im[:, margin:-margin, margin:-margin, :]
    else:
        return im[:, margin:-margin, margin:-margin]


def rgb_augment(im, rndm=True, shuff_ind=0):
    if rndm:
        shuffle = sample(range(3), 3)
    else:
        shuffle = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        shuffle = shuffle[shuff_ind]

    return im[:, :, shuffle]


def probability_map(im, crop_size):
    # margin of probabilities that will be zero
    margin = crop_size // 2 - 1
    prob_map = np.zeros(im.shape[0:2])
    # Gradient calculation
    gx, gy, _ = np.gradient(im)
    grad_magnitude = np.sum(np.sqrt(gx ** 2 + gy ** 2), axis=2)
    # Convolving with rect to get a map of probabilities per crop
    rect = np.ones([crop_size - 3, crop_size - 3])
    grad_magnitude_conv = convolve2d(grad_magnitude, rect, 'same')
    # Copying the values without the margins of the image
    prob_map[margin:-margin, margin:-margin] = grad_magnitude_conv[margin:-margin, margin:-margin]
    # normalize for probabilities
    sum_of_grads = np.sum(prob_map)
    prob_map = prob_map / sum_of_grads

    return prob_map


def choose_center_of_crop(prob_map):
    # Retrieving a probability map and reshaping to be a vector
    prob_vector = np.reshape(prob_map, prob_map.shape[0] * prob_map.shape[1])
    # creating a vector of indices to match the image
    indices = np.arange(start=0, stop=prob_map.shape[0] * prob_map.shape[1])
    # Choosing an index according to the probabilities
    index_choice = np.random.choice(indices, p=prob_vector)
    # Translating to an index in the image - row, column
    return index_choice // prob_map.shape[1], index_choice % prob_map.shape[1]


def create_loss_map(im, window=5, clip_rng=np.array([0.0, 255.0])):
    # Counting number of pixels for normalization issues
    numel = im.shape[0] * im.shape[1]
    # rgb2gray if image is in color
    gray = np.dot(im[:, :, 0:3], [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im
    # Gradient calculation
    gx, gy = np.gradient(gray)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    processed_gmag = convolve2d(gmag, np.ones(shape=(window, window)), 'same')
    # pad the gmag with zeros the size of the process to eliminate artifacts
    margin = int((window + window % 2) / 2)
    loss_map = np.zeros_like(processed_gmag)
    # ignoring edges + clipping
    loss_map[margin:-margin, margin:-margin] = np.clip(processed_gmag[margin:-margin, margin:-margin], clip_rng[0], clip_rng[1])
    # Normalizing the grad magnitude to sum to numel
    norm_factor = np.sum(loss_map) / numel
    loss_map = loss_map / norm_factor

    # In case the image is color, return 3 channels with the loss map duplicated
    if len(im.shape) == 3:
        loss_map = np.expand_dims(loss_map, axis=2)
        loss_map = np.append(np.append(loss_map, loss_map, axis=2), loss_map, axis=2)

    return loss_map


def image_float2int(im):
    """converts a float image to uint"""
    if np.max(im) < 2:
        im = im * 255.
    return np.uint8(im)


def image_int2float(im):
    """converts a uint image to float"""
    return np.float32(im) / 255. if np.max(im) > 2 else im


def back_project_image(lr, sf=2, output_shape=None, down_kernel='cubic', up_kernel='cubic', bp_iters=8):
    """Runs 'bp_iters' iteration of back projection SR technique"""
    tmp_sr = imresize(lr, scale_factor=sf, output_shape=output_shape, kernel=up_kernel)
    for _ in range(bp_iters):
        tmp_sr = back_projection(y_sr=tmp_sr, y_lr=lr, down_kernel=down_kernel, up_kernel=up_kernel, sf=sf)
    return tmp_sr


def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None):
    """Projects the error between the downscaled SR image and the LR image"""
    y_sr += imresize(y_lr - imresize(y_sr,
                                     scale_factor=1.0 / sf,
                                     output_shape=y_lr.shape,
                                     kernel=down_kernel),
                     scale_factor=sf,
                     output_shape=y_sr.shape,
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)
