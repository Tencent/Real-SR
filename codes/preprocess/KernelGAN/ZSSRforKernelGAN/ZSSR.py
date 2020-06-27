import matplotlib.image as img
from ZSSRforKernelGAN.zssr_configs import Config
from ZSSRforKernelGAN.zssr_utils import *
import numpy as np
import tensorflow as tf


class ZSSR:
    # Basic current state variables initialization / declaration
    kernel = None
    learning_rate = None
    hr_father = None
    lr_son = None
    sr = None
    sf = None
    gt_per_sf = None
    final_sr = None
    hr_fathers_sources = []

    # Output variables initialization / declaration
    reconstruct_output = None
    train_output = None
    output_shape = None

    # Counters and logs initialization
    iter = 0
    base_sf = 1.0
    base_ind = 0
    sf_ind = 0
    mse = []
    mse_rec = []
    interp_rec_mse = []
    interp_mse = []
    mse_steps = []
    loss = []
    learning_rate_change_iter_nums = []
    fig = None

    # Network tensors (all tensors end with _t to distinguish)
    learning_rate_t = None
    lr_son_t = None
    hr_father_t = None
    filters_t = None
    layers_t = None
    net_output_t = None
    loss_t = None
    loss_map_t = None
    train_op = None
    init_op = None

    # Parameters related to plotting and graphics
    plots = None
    loss_plot_space = None
    lr_son_image_space = None
    hr_father_image_space = None
    out_image_space = None

    # A map representing the gradient magnitude of the image at every crop
    prob_map = None
    cropped_loss_map = None
    avg_grad = 1
    loss_map = []
    loss_map_sources = []

    # Tensorflow graph default
    sess = None

    def __init__(self, input_img_path, scale_factor=2, kernels=None, is_real_img=False, noise_scale=1.):
        # Acquire meta parameters configuration from configuration class as a class variable
        self.conf = Config(scale_factor, is_real_img, noise_scale)
        # Read input image
        self.input = img.imread(input_img_path)
        # Discard the alpha channel from images
        if self.input.shape[-1] == 4:
            self.input = img.imread(input_img_path)[:, :, :3]
        # For gray-scale images - add a 3rd dimension to fit the network
        elif len(self.input.shape) == 2:
            self.input = np.expand_dims(self.input, -1)
        self.input = self.input / 255. if self.input.dtype == 'uint8' else self.input
        self.gt = None
        # Shift kernel to avoid misalignment
        self.kernels = [kernel_shift(kernel, sf) for kernel, sf in zip(kernels, self.conf.scale_factors)] if kernels is not None else [self.conf.downscale_method] * len(self.conf.scale_factors)

        # Prepare TF default computational graph
        self.model = tf.Graph()

        # Build network computational graph
        self.build_network(self.conf)

        # Initialize network weights and meta parameters
        self.init_sess(init_weights=True)

        # The first hr father source is the input (source goes through augmentation to become a father)
        # Later on, if we use gradual sr increments, results for intermediate scales will be added as sources.
        self.hr_fathers_sources = [self.input]

        # Create a loss map reflecting the weights per pixel of the image
        self.loss_map = create_loss_map(im=self.input) if self.conf.grad_based_loss_map else np.ones_like(self.input)

        # loss maps that correspond to the father sources array
        self.loss_map_sources = [self.loss_map]

    def run(self):
        # Run gradually on all scale factors (if only one jump then this loop only happens once)
        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):
            # Relative_sf (used when base change is enabled. this is when input is the output of some previous scale)
            sf = [sf, sf] if np.isscalar(sf) else sf
            self.sf = np.array(sf) / np.array(self.base_sf)

            self.output_shape = np.uint(np.ceil(np.array(self.input.shape[0:2]) * sf))

            # Initialize network
            self.init_sess(init_weights=self.conf.init_net_for_each_sf)

            # Train the network
            self.train()

            # Use augmented outputs and back projection to enhance result. Also save the result.
            post_processed_output = self.final_test()

            # Keep the results for the next scale factors SR to use as dataset
            self.hr_fathers_sources.append(post_processed_output)

            # append a corresponding map loss
            self.loss_map_sources.append(create_loss_map(im=post_processed_output)) if self.conf.grad_based_loss_map else self.loss_map_sources.append(np.ones_like(post_processed_output))

            # In some cases, the current output becomes the new input. If indicated and if this is the right scale to
            # become the new base input. all of these conditions are checked inside the function.
            self.base_change()

        # Return the final post processed output.
        # noinspection PyUnboundLocalVariable
        return post_processed_output

    def build_network(self, meta):
        with self.model.as_default():
            # Learning rate tensor
            self.learning_rate_t = tf.placeholder(tf.float32, name='learning_rate')

            # Input image
            self.lr_son_t = tf.placeholder(tf.float32, name='lr_son')

            # Ground truth (supervision)
            self.hr_father_t = tf.placeholder(tf.float32, name='hr_father')

            # Loss map
            self.loss_map_t = tf.placeholder(tf.float32, name='loss_map')

            # Filters
            self.filters_t = [tf.get_variable(shape=meta.filter_shape[ind], name='filter_%d' % ind,
                                              initializer=tf.random_normal_initializer(
                                                  stddev=np.sqrt(meta.init_variance / np.prod(
                                                      meta.filter_shape[ind][0:3]))))
                              for ind in range(meta.depth)]

            # Activate filters on layers one by one (this is just building the graph, no calculation is done here)
            self.layers_t = [self.lr_son_t] + [None] * meta.depth
            for l in range(meta.depth - 1):
                self.layers_t[l + 1] = tf.nn.relu(tf.nn.conv2d(self.layers_t[l], self.filters_t[l], [1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1)))

            # Last conv layer (Separate because no ReLU here)
            l = meta.depth - 1
            self.layers_t[-1] = tf.nn.conv2d(self.layers_t[l], self.filters_t[l], [1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1))

            # Output image (Add last conv layer result to input, residual learning with global skip connection)
            self.net_output_t = self.layers_t[-1] + self.conf.learn_residual * self.lr_son_t

            # Final loss (L1 loss between label and output layer)
            self.loss_t = tf.reduce_mean(tf.reshape(tf.abs(self.net_output_t - self.hr_father_t) * self.loss_map_t, [-1]))

            # Apply adam optimizer
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t).minimize(self.loss_t)
            # self.init_op = tf.initialize_all_variables()
            self.init_op = tf.global_variables_initializer()

    def init_sess(self, init_weights=True):
        # Sometimes we only want to initialize some meta-params but keep the weights as they were
        if init_weights:
            # These are for GPU consumption, preventing TF to catch all available GPUs
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            # Initialize computational graph session
            self.sess = tf.Session(graph=self.model, config=config)

            # Initialize weights
            self.sess.run(self.init_op)

        # Initialize all counters etc
        self.loss = [None] * self.conf.max_iters
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.iter = 0
        self.learning_rate = self.conf.learning_rate
        self.learning_rate_change_iter_nums = [0]

    def forward_backward_pass(self, lr_son, hr_father, cropped_loss_map):
        # First gate for the lr-son into the network is interpolation to the size of the father
        # Note: we specify both output_size and scale_factor. best explained by example: say father size is 9 and sf=2,
        # small_son size is 4. if we upscale by sf=2 we get wrong size, if we upscale to size 9 we get wrong sf.
        # The current imresize implementation supports specifying both.
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method)
        # Create feed dict
        feed_dict = {'learning_rate:0': self.learning_rate,
                     'lr_son:0': np.expand_dims(interpolated_lr_son, 0),
                     'hr_father:0': np.expand_dims(hr_father, 0),
                     'loss_map:0': np.expand_dims(cropped_loss_map, 0)}

        # Run network
        _, self.loss[self.iter], train_output = self.sess.run([self.train_op, self.loss_t, self.net_output_t],
                                                              feed_dict)
        return np.clip(np.squeeze(train_output), 0, 1)

    def forward_pass(self, lr_son, hr_father_shape=None):
        # First gate for the lr-son into the network is interpolation to the size of the father
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father_shape, self.conf.upscale_method)

        # Create feed dict
        feed_dict = {'lr_son:0': np.expand_dims(interpolated_lr_son, 0)}

        # Run network
        return np.clip(np.squeeze(self.sess.run([self.net_output_t], feed_dict)), 0, 1)

    def learning_rate_policy(self):
        # fit linear curve and check slope to determine whether to do nothing, reduce learning rate or finish
        if (not (1 + self.iter) % self.conf.learning_rate_policy_check_every
                and self.iter - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters):
            # noinspection PyTupleAssignmentBalance
            [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-int(self.conf.learning_rate_slope_range /
                                                                       self.conf.run_test_every):],
                                                   self.mse_rec[-int(self.conf.learning_rate_slope_range /
                                                                     self.conf.run_test_every):],
                                                   1, cov=True)

            # We take the the standard deviation as a measure
            std = np.sqrt(var)

            # Determine learning rate maintaining or reduction by the ration between slope and noise
            if -self.conf.learning_rate_change_ratio * slope < std:
                self.learning_rate /= 10

                # Keep track of learning rate changes for plotting purposes
                self.learning_rate_change_iter_nums.append(self.iter)

    def quick_test(self):
        # There are four evaluations needed to be calculated:

        # 1. True MSE (only if ground-truth was given), note: this error is before post-processing.
        # Run net on the input to get the output super-resolution (almost final result, only post-processing needed)
        self.sr = self.forward_pass(self.input)
        self.mse = (self.mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - self.sr)))]
                    if self.gt_per_sf is not None else None)

        # 2. Reconstruction MSE, run for reconstruction- try to reconstruct the input from a downscaled version of it
        self.reconstruct_output = self.forward_pass(self.father_to_son(self.input), self.input.shape)
        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.input - self.reconstruct_output))))

        # 3. True MSE of simple interpolation for reference (only if ground-truth was given)
        if self.gt_per_sf is not None:
            interp_sr = imresize(self.input, self.sf, self.output_shape, self.conf.upscale_method)

            self.interp_mse = self.interp_mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - interp_sr)))]
        else:
            self.interp_mse = None

        # 4. Reconstruction MSE of simple interpolation over downscaled input
        interp_rec = imresize(self.father_to_son(self.input), self.sf, self.input.shape[:], self.conf.upscale_method)

        self.interp_rec_mse.append(np.mean(np.ndarray.flatten(np.square(self.input - interp_rec))))

        # Track the iters in which tests are made for the graphics x axis
        self.mse_steps.append(self.iter)

    def train(self):
        # main training loop
        for self.iter in range(self.conf.max_iters):
            # Use augmentation from original input image to create current father.
            # If other scale factors were applied before, their result is also used (hr_fathers_in)
            # crop_center = choose_center_of_crop(self.prob_map) if self.conf.choose_varying_crop else None
            crop_center = None

            self.hr_father, self.cropped_loss_map = \
                random_augment(ims=self.hr_fathers_sources,
                               base_scales=[1.0] + self.conf.scale_factors,
                               leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                               no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                               min_scale=self.conf.augment_min_scale,
                               max_scale=([1.0] + self.conf.scale_factors)[len(self.hr_fathers_sources) - 1],
                               allow_rotation=self.conf.augment_allow_rotation,
                               scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                               shear_sigma=self.conf.augment_shear_sigma,
                               crop_size=self.conf.crop_size,
                               allow_scale_in_no_interp=self.conf.allow_scale_in_no_interp,
                               crop_center=crop_center,
                               loss_map_sources=self.loss_map_sources)

            # Get lr-son from hr-father
            self.lr_son = self.father_to_son(self.hr_father)
            # run network forward and back propagation, one iteration (This is the heart of the training)
            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, self.cropped_loss_map)

            # Test network
            if self.conf.run_test and (not self.iter % self.conf.run_test_every):
                self.quick_test()

            # Consider changing learning rate or stop according to iteration number and losses slope
            self.learning_rate_policy()

            # stop when minimum learning rate was passed
            if self.learning_rate < self.conf.min_learning_rate:
                break

    def father_to_son(self, hr_father):
        # Create son out of the father by downscaling and if indicated adding noise
        lr_son = imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    def final_test(self):
        # Run over 8 augmentations of input - 4 rotations and mirror (geometric self ensemble)
        # The weird range means we only do it once if output_flip is disabled
        # We need to check if scale factor is symmetric to all dimensions, if not we will do 180 jumps rather than 90

        outputs = []
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
            # Rotate 90*k degrees & mirror flip when k>=4
            test_input = np.rot90(self.input, k) if k < 4 else np.fliplr(np.rot90(self.input, k))

            # Apply network on the rotated input
            tmp_output = self.forward_pass(test_input)

            # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)

            # fix SR output with back projection technique for each augmentation
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(tmp_output, self.input, down_kernel=self.kernel,
                                             up_kernel=self.conf.upscale_method, sf=self.sf)

            # save outputs from all augmentations
            outputs.append(tmp_output)

            # Take the median over all 8 outputs
            almost_final_sr = np.median(outputs, 0)

            # Again back projection for the final fused result
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                almost_final_sr = back_projection(almost_final_sr, self.input, down_kernel=self.kernel,
                                                  up_kernel=self.conf.upscale_method, sf=self.sf)

        # Now we can keep the final result (in grayscale case, colors still need to be added, but we don't care
        # because it is done before saving and for every other purpose we use this result)
        # noinspection PyUnboundLocalVariable
        self.final_sr = almost_final_sr

        # Add colors to result image in case net was activated only on grayscale
        return self.final_sr

    def base_change(self):
        # If there is no base scale large than the current one get out of here
        if len(self.conf.base_change_sfs) < self.base_ind + 1:
            return

        # Change base input image if required (this means current output becomes the new input)
        if abs(self.conf.scale_factors[self.sf_ind] - self.conf.base_change_sfs[self.base_ind]) < 0.001:
            if len(self.conf.base_change_sfs) > self.base_ind:
                # The new input is the current output
                self.input = self.final_sr

                # The new base scale_factor
                self.base_sf = self.conf.base_change_sfs[self.base_ind]

                # Keeping track- this is the index inside the base scales list (provided in the config)
                self.base_ind += 1

            print('base changed to %.2f' % self.base_sf)
