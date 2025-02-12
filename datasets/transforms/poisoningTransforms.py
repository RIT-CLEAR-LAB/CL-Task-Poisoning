import numpy as np
import cv2


class DefocusBlur:
    def __init__(self, severity=1):
        self.severity = severity

    def __call__(self, x):
        return self.defocus_blur(x, self.severity)

    @staticmethod
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    def defocus_blur(self, x, severity):
        c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

        x = np.array(x) / 255.0
        kernel = self.disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

        x = np.clip(channels, 0, 1) * 255.0
        return x.astype(np.uint8)


class GaussianNoise:
    def __init__(self, severity=1):
        self.severity = severity

    def __call__(self, x):
        return self.gaussian_noise(x, self.severity)

    @staticmethod
    def gaussian_noise(x, severity):
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]

        x = np.array(x) / 255.0
        x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255.0
        x = x.astype(np.uint8)
        return x


class ShotNoise:
    def __init__(self, severity=1):
        self.severity = severity

    def __call__(self, x):
        return self.shot_noise(x, self.severity)

    @staticmethod
    def shot_noise(x, severity):
        c = [500, 250, 100, 75, 50][severity - 1]

        x = np.array(x) / 255.0
        x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255.0
        return x.astype(np.uint8)


class SpeckleNoise:
    def __init__(self, severity=1):
        self.severity = severity

    def __call__(self, x):
        return self.speckle_noise(x, self.severity)

    @staticmethod
    def speckle_noise(x, severity):
        c = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]

        x = np.array(x) / 255.0
        x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255.0
        return x.astype(np.uint8)


class PixelPermutation:
    def __init__(self, severity=1):
        self.severity = severity

    def __call__(self, x):
        return self.permute_pixels(x, self.severity)

    @staticmethod
    def permute_pixels(x, severity):
        proportions = [0.1, 0.2, 0.3, 0.4, 0.5]
        shuffle_proportion = proportions[severity - 1]

        x = np.array(x)
        # keeping channels separate: (H * W, 3)
        x_flat = np.array(x).reshape(-1, x.shape[-1])
        num_pixels = x_flat.shape[0]
        num_shuffle = int(num_pixels * shuffle_proportion)

        indices = np.arange(num_pixels)
        shuffle_indices = np.random.choice(indices, size=num_shuffle, replace=False)
        permuted_indices = np.random.permutation(shuffle_indices)
        x_flat[shuffle_indices] = x_flat[permuted_indices]

        return x_flat.reshape(x.shape)


class BackdoorAttack:
    def __init__(self, square_size=2, color=[1.0, 0.0, 0.0], position=(0, 0)):
        if not all(0 <= c <= 1 for c in color):
            raise ValueError("Color values must be between 0 and 1")
        self.square_size = square_size
        self.color = color
        self.position = position

    def __call__(self, img):
        return self.add_trigger(img, self.square_size, self.color, self.position)

    @staticmethod
    def add_trigger(img, square_size, color, position):
        modified_image = np.array(img) / 255.0
        h, w, c = modified_image.shape

        assert (
            len(color) == c
        ), "Color array must have the same number of channels as the input image"
        assert (
            position[0] + square_size <= w or position[1] + square_size <= h
        ), "Square cannot exceed image dimensions"

        x, y = position
        modified_image[y : y + square_size, x : x + square_size, :] = color

        modified_image = np.clip(modified_image, 0, 1) * 255.0
        return modified_image.astype(np.uint8)


class Identity:
    def __init__(self, severity=1):
        pass

    def __call__(self, x):
        x = np.array(x)
        x = x.astype(np.uint8)
        return x
