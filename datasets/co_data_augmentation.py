from datasets.transforms.co_transform import *

class CoDataAugmentation(object):
    def __init__(self, mean_RGB, std_RGB, mean_IR, std_IR) -> None:
        self.transforms = CoTransCompose([
            CoRandomCrop(ratio=(0.4, 1)),
            CoResize([224, 224]),
            CoRandomHoreizontalFlip(0.5),
            CoToTensor(),
            CoNormalize(mean_RGB, std_RGB, mean_IR, std_IR)
        ])
    
    def __call__(self,image1,image2):
        image1, image2 = self.transforms(image1,image2)
        return image1, image2