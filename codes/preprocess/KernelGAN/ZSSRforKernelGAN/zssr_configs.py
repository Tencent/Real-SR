import os


class Config:
    # Network meta params
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    scale_factors = [[2.0, 2.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
    base_change_sfs = []  # list of scales after which the input is changed to be the output (recommended for high sfs)
    max_iters = 3000
    min_iters = 256
    min_learning_rate = 9e-6  # this tells the algorithm when to stop (specify lower than the last learning-rate)
    width = 64
    depth = 8
    # network meta params that by default are determined (by other params) by other params but can be changed
    filter_shape = ([[3, 3, 3, width]] + [[3, 3, width, width]] * (depth - 2) + [[3, 3, width, 3]])

    output_flip = False  # changed from True  # geometric self-ensemble (see paper)
    downscale_method = 'cubic'  # a string ('cubic', 'linear'...), has no meaning if kernel given
    upscale_method = 'cubic'  # this is the base interpolation from which we learn the residual (same options as above)
    downscale_gt_method = 'cubic'  # when ground-truth given and intermediate scales tested, we shrink gt to wanted size
    learn_residual = True  # when true, we only learn the residual from base interpolation
    init_variance = 0.1  # variance of weight initializations, typically smaller when residual learning is on
    back_projection_iters = [2]  # for each scale num of bp iterations (same length as scale_factors)
    random_crop = True
    crop_size = 128
    noise_std = 0.  # adding noise to lr-sons. small for real images, bigger for noisy images and zero for ideal case
    init_net_for_each_sf = False  # for gradual sr- should we optimize from the last sf or initialize each time?

    # Params concerning learning rate policy
    learning_rate = 0.001
    learning_rate_change_ratio = 1.5  # ratio between STD and slope of linear fit, under which lr is reduced
    learning_rate_policy_check_every = 60
    learning_rate_slope_range = 256

    # Data augmentation related params
    augment_leave_as_is_probability = 1  # changed from 0.05
    augment_no_interpolate_probability = 0  # changed from 0.45
    augment_min_scale = 0  # changed from 0.5
    augment_scale_diff_sigma = 0  # changed from 0.25
    augment_shear_sigma = 0  # changed from 0.1
    augment_allow_rotation = False  # changed from True  # recommended false for non-symmetric kernels

    # params related to test and display
    run_test = True
    run_test_every = 50

    allow_scale_in_no_interp = False
    grad_based_loss_map = True  # In the case a loss should be calculated w.r.t gradient map

    def __init__(self, scale_factor, is_real_img, noise_scale):
        self.scale_factors = [[scale_factor, scale_factor]] if type(scale_factor) is int else scale_factor
        if is_real_img:
            print('\nZSSR configuration is for a real image')
            self.back_projection_iters = [0]  # no B.P
            self.noise_std = 0.0125 * noise_scale  # Add noise to sons
        if type(self.scale_factors[0]) is list:  # for gradual SR
            self.back_projection_iters = [self.back_projection_iters[0], self.back_projection_iters[0]]

