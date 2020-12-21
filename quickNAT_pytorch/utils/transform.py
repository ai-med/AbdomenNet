import numpy as np
from skimage.transform import rescale, rotate
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import unsharp_mask
from skimage.exposure import adjust_gamma, rescale_intensity
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import Compose


def transforms(unsharp_prob=None,
               rescale_prob=None,
               gamma_prob=None,
               flip_prob=None,
               scale_prob=None,
               rotate_prob=None,
               deform_prob=None,
               denoise_prob=None,
               ):

    transform_list = []
    if unsharp_prob is not None:
        transform_list.append(UnsharpMasking(prob=unsharp_prob))
    if rescale_prob is not None:
        transform_list.append(RescaleIntensity(prob=rescale_prob))
    if gamma_prob is not None:
        transform_list.append(GammaCorrection(prob=gamma_prob))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(prob=flip_prob))
    if scale_prob is not None:
        transform_list.append(Scale(prob=scale_prob))
    if rotate_prob is not None:
        transform_list.append(Rotate(prob=rotate_prob))
    if deform_prob is not None:
        transform_list.append(ElasticTransform(prob=deform_prob))
    if denoise_prob is not None:
        transform_list.append(TVDenoising(prob=denoise_prob))

    return Compose(transform_list)


class Scale(object):

    def __init__(self,
                 scale=0.05,
                 prob=0.5,
                 ):
        self.scale = scale
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(image, scale, multichannel=True, preserve_range=True, mode="constant", anti_aliasing=False)
        mask = rescale(
            mask, scale, order=0, multichannel=False, preserve_range=True, mode="constant", anti_aliasing=False)

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding[0:2], mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self,
                 angle=10,
                 prob=0.5,
                 ):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )

        return image, mask


class HorizontalFlip(object):

    def __init__(self,
                 prob=0.5,
                 ):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask


class TVDenoising(object):

    def __init__(self,
                 denoise_weight=0.1,
                 prob=0.5
                 ):
        self.denoise_weight = denoise_weight
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        weight = np.random.uniform(low=0, high=self.denoise_weight)
        image = denoise_tv_chambolle(image, weight=weight, multichannel=True)

        return image, mask


class UnsharpMasking(object):

    def __init__(self,
                 sharping_magnitude=(0, 2),
                 prob=0.5,
                 ):
        self.sharping_magnitude = sharping_magnitude
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        amount = np.random.uniform(low=self.sharping_magnitude[0], high=self.sharping_magnitude[1])
        image = unsharp_mask(image, amount=amount)

        return image, mask


class ElasticTransform(object):

    def __init__(self,
                 alpha=(0, 300),
                 sigma=(10, 12),
                 prob=0.5,
                 ):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        alpha = np.random.uniform(low=self.alpha[0], high=self.alpha[1])
        sigma = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
        random_state = np.random.RandomState(None)

        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if len(image.shape) > 2:
            c = image.shape[2]
            distorted_image = [map_coordinates(image[:, :, i], indices, order=1, mode='reflect') for i in range(c)]
            distorted_image = np.concatenate(distorted_image, axis=1)
        else:
            distorted_image = map_coordinates(image, indices, order=1, mode='reflect')

        distorted_mask = map_coordinates(mask, indices, order=1, mode='reflect')

        image = distorted_image.reshape(image.shape)
        mask = distorted_mask.reshape(mask.shape)

        return image, mask


# it's relatively slow, i.e. ~1s per patch of size 64x200x200
# For 3D data.
class ElasticDeformation:
    """
    code from: https://github.com/wolny/pytorch-3dunet/blob/master/augment/transforms.py
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """
    def __init__(self, spline_order, alpha=32, sigma=4):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smothing factor for Gaussian filter
        """
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, images):
        #assert len(images) == 2
        img = images[0]
        target = images[1]
        assert img.ndim == 3
        dz = gaussian_filter(np.random.randn(*img.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter(np.random.randn(*img.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dx = gaussian_filter(np.random.randn(*img.shape), self.sigma, mode="constant", cval=0) * self.alpha
        z_dim, y_dim, x_dim = img.shape
        z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
        indices = z + dz, y + dy, x + dx
        img = map_coordinates(img, indices, order=self.spline_order, mode='reflect')
        target = map_coordinates(target, indices, order=0, mode='reflect')
        return [img, target]


class GammaCorrection(object):

    def __init__(self,
                 gamma_value=(0.3, 1.5),
                 prob=0.5,
                 ):
        self.gamma_value = gamma_value
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        gamma = np.random.uniform(low=self.gamma_value[0], high=self.gamma_value[1])
        image = adjust_gamma(image, gamma=gamma)

        return image, mask


class RescaleIntensity(object):

    def __init__(self,
                 shift_factor=(-0.1, 0.1),
                 prob=0.5,
                 ):
        self.shift_factor = shift_factor
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.prob:
            return image, mask

        scale_factor = np.random.uniform(low=self.shift_factor[0], high=self.shift_factor[1])
        img_min, img_max = np.amin(image, axis=(0, 1)), np.amax(image, axis=(0, 1))
        image[:, :, 0] = rescale_intensity(image[:, :, 0], in_range=(img_min[0], img_max[0] + scale_factor),
                                           out_range=(img_min[0], img_max[0] + scale_factor))
        image[:, :, 1] = rescale_intensity(image[:, :, 1], in_range=(img_min[1], img_max[1] + scale_factor),
                                           out_range=(img_min[1], img_max[1] + scale_factor))

        return image, mask
