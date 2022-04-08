import random
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        return image


class RandomRotate90(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.rotate(image, 90)
        return image


class RandomMinorRotate(object):
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, image):
        rotate_angle = random.randint(-self.max_angle, self.max_angle)
        image = F.rotate(image, rotate_angle)
        return image


class GaussianBlur(object):
    def __init__(self, kernel, sigma):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, image):
        image = F.gaussian_blur(image, kernel_size=self.kernel, sigma=self.sigma)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
        return image


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = pad_if_smaller(image, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        return image


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = F.center_crop(image, self.size)
        return image


class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
