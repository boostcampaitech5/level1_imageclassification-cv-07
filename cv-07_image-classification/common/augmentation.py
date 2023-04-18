import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter
import albumentations as A


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.Resize(height=resize[0], width=resize[1]),
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            A.pytorch.ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform(image=image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.CenterCrop(320, 256),
            A.Resize(height=resize[0], width=resize[1]),
            A.ColorJitter(0.1, 0.1, 0.1, 0.1),
            A.Normalize(mean=mean, std=std),
            A.pytorch.ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform(image=image)