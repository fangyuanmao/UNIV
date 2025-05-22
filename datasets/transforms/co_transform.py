
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class CoRandomHoreizontalFlip(object):
    def __init__(self , flip_prob):
        self.flip_prob = flip_prob
    
    def __call__(self,image1,image2):
        if random.random() < self.flip_prob:
            image1 = F.hflip(image1)
            image2 = F.hflip(image2)
        return image1, image2
    
class CoCenterCrop(object):
    def __init__(self, size) -> None:
        self.size = size
    
    def __call__(self, image1, image2):
        image1 = F.center_crop(image1, self.size)
        image2 = F.center_crop(image2, self.size)
        return image1, image2

class CoRandomCrop(object):
    def __init__(self, ratio=(0.4, 1)) -> None:
        self.ratio = ratio
    
    def __call__(self, image1: Image.Image, image2: Image.Image):
        width, height = image1.size
        
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        crop_size = (int(height * ratio), int(width * ratio))
        
        i = random.randint(0, height - crop_size[0])
        j = random.randint(0, width - crop_size[1])

        image1 = F.crop(image1, i, j, crop_size[0], crop_size[1])
        image2 = F.crop(image2, i, j, crop_size[0], crop_size[1])
        
        return image1, image2

class CoNormalize(object):
    def __init__(self,mean1:tuple,std1:tuple,mean2:tuple,std2:tuple) -> None:
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
    
    def __call__(self,image1,image2):
        image1 = F.normalize(image1,mean=self.mean1,std=self.std1)
        image2 = F.normalize(image2,mean=self.mean2,std=self.std2)
        return image1,image2

class CoResize(object):
    def __init__(self,size:list) -> None:
        self.size = size
    
    def __call__(self,image1,image2):
        image1 = F.resize(image1,self.size)
        image2 = F.resize(image2,self.size)
        return image1, image2

class CoToTensor(object):
    def __init__(self) -> None:
        pass
    
    def __call__(self,image1,image2):
        image1 = F.to_tensor(image1)
        image2 = F.to_tensor(image2)
        return image1, image2

class CoTransCompose(object):
    def __init__(self,transforms:list) -> None:
        self.transforms = transforms

    def __call__(self,image1,image2):
        for t in self.transforms:
            image1, image2 = t(image1,image2)
        return image1, image2
    