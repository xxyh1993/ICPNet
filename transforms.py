import torchvision.transforms.functional as F
import numbers
import random
from PIL import Image
# import collections
import math
import torch
from torchvision.utils import _log_api_usage_once
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        # >>> transforms.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms, flag=1):
        self.transforms = transforms
        self.flag = flag

    def __call__(self, img):
        if isinstance(img, list):
            i=0
            for t in self.transforms:
                img = t(img)
                if i == (len(self.transforms)-2):
                    img_ori = img[0]
                i += 1
            return img[0], img[1] 
        else:
            i=0
            for t in self.transforms:
                img = t(img)                
                if i == (len(self.transforms)-2):
                    img_ori = img
                i += 1
            if self.flag == 1:
                return img  #  img_ori is returned for visualization, if need, flag=0, or flg=1
            elif self.flag == 0:
                return img, img_ori

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        # return {'images': F.to_tensor(sample['images']), 'labels': F.to_tensor(sample['labels'])}
        if isinstance(sample, list):
            for i in range(len(sample)):
                if isinstance(sample[1], str):
                    sample[i] = F.to_tensor(sample[i])
                    break
                else:
                    sample[i] = F.to_tensor(sample[i])
        else:
            sample = F.to_tensor(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(sample, list):
            for xy in range(len(sample)):
                if isinstance(sample[1], str):
                    sample[xy] = F.normalize(sample[xy], self.mean[xy], self.std[xy])
                    break
                else:
                    sample[xy] = F.normalize(sample[xy], self.mean[xy], self.std[xy])
        else:
            sample = F.normalize(sample, self.mean[0], self.std[0])
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types. See also below the ``antialias`` parameter, which can help making the output of PIL images and tensors
        closer.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e. the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.
    """
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias="warn"):
        super().__init__()
        _log_api_usage_once(self)
        self.size = size
        self.max_size = max_size

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled. According to icpnet_train_classifier_mnist,
            the shape of img is [imagergb, edge_label], whereas the size of edge_label is 224*224

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(img, list):
            img[0] = F.resize(img[0], self.size, self.interpolation, self.max_size, self.antialias)
        else:
            img = F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)

        return  img

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if isinstance(img, list):
                img[0] = F.hflip(img[0])
                img[1] = F.hflip(img[1])
                return img
            else:
                return F.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.NEAREST):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        if isinstance(sample, list):
            i, j, h, w = self.get_params(sample[0], self.scale, self.ratio)
            sample[0] = F.resized_crop(sample[0], i, j, h, w, self.size, Image.BILINEAR)
            if isinstance(sample[1], str):
                pass
            else:
                sample[1] = F.resized_crop(sample[1], i, j, h, w, self.size, Image.BILINEAR)
            for xy in range(len(sample)):
                if xy >= 2:
                    sample[xy] = F.resized_crop(sample[xy], i, j, h, w, self.size, Image.BILINEAR)
        else:
            i, j, h, w = self.get_params(sample, self.scale, self.ratio)
            sample = F.resized_crop(sample, i, j, h, w, self.size, Image.BILINEAR)
        return sample
        # return {'images': F.resized_crop(sample['images'], i, j, h, w, self.size, Image.BILINEAR),
        #  'labels': F.resized_crop(sample['labels'], i, j, h, w, self.size, Image.NEAREST)}

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomCrop(object):
    @staticmethod
    def get_params(img, output_size):
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            if isinstance(img, list):
                _, height, width = F.get_dimensions(img[0])
                # pad the width if needed
                if self.pad_if_needed and width < self.size[1]:
                    padding = [self.size[1] - width, 0]
                    img = F.pad(img, padding, self.fill, self.padding_mode)
                # pad the height if needed
                if self.pad_if_needed and height < self.size[0]:
                    padding = [0, self.size[0] - height]
                    img = F.pad(img, padding, self.fill, self.padding_mode)

                i, j, h, w = self.get_params(img[0], self.size)
                for index in range(len(img)):
                    if isinstance(img[index], str):
                        break
                    img[index] = F.crop(img[index], i, j, h, w)
            else:
                _, height, width = F.get_dimensions(img)
                # pad the width if needed
                if self.pad_if_needed and width < self.size[1]:
                    padding = [self.size[1] - width, 0]
                    img = F.pad(img, padding, self.fill, self.padding_mode)
                # pad the height if needed
                if self.pad_if_needed and height < self.size[0]:
                    padding = [0, self.size[0] - height]
                    img = F.pad(img, padding, self.fill, self.padding_mode)

                i, j, h, w = self.get_params(img, self.size)
                img = F.crop(img, i, j, h, w)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"
