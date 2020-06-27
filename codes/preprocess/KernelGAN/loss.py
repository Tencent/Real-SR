import torch
import torch.nn as nn
from torch.autograd import Variable
from util import shave_a2b, resize_tensor_w_kernel, create_penalty_mask, map2tensor


# noinspection PyUnresolvedReferences
class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.L1Loss(reduction='mean')
        # Make a shape
        d_last_layer_shape = [1, 1, d_last_layer_size, d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        # Compute the loss
        return self.loss(d_last_layer, label_tensor)


class DownScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]
        self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output):
        downscaled = resize_tensor_w_kernel(im_t=g_input, k=self.bicubic_kernel, sf=self.scale_factor)
        # Shave the downscaled to fit g_output
        return self.loss(g_output, shave_a2b(downscaled, g_output))


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))
