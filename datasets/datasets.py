import os
from PIL import Image
from torch.utils.data import Dataset
from datasets.transforms.co_transform import CoTransCompose

class RGB_pair_IR_dataset(Dataset):
    def __init__(self, 
                 transform:CoTransCompose, 
                 data_path=['/public/home/public/IRPretrainDataset/FLIR-ALIGN',
                            '/public/home/public/IRPretrainDataset/KAIST_MMSEG',
                            '/public/home/public/IRPretrainDataset/LasHeR0428_rename',
                            '/public/home/public/IRPretrainDataset/LLVIP',
                            '/public/home/public/IRPretrainDataset/VisDrone']
                 ):
        super().__init__()
        self.rgb = []
        self.inf = []
        
        for i in range(len(data_path)):
            rgb_data_path = os.path.join(data_path[i], 'vi_images/training/')
            rgb = sorted(os.listdir(rgb_data_path))
            rgb = [os.path.join(rgb_data_path, file_name) for file_name in rgb]
            self.rgb = self.rgb + rgb

            inf_data_path = os.path.join(data_path[i], 'images/training/')
            inf = sorted(os.listdir(inf_data_path))
            inf = [os.path.join(inf_data_path, file_name) for file_name in inf]
            self.inf = self.inf + inf

        for rgb_path, inf_path in zip(self.rgb, self.inf):
            rgb_filename = os.path.basename(rgb_path)
            inf_filename = os.path.basename(inf_path)
            if rgb_filename != inf_filename:
                print(f"Mismatch detected!")
                print(f"RGB filename: {rgb_filename}")
                print(f"INF filename: {inf_filename}")
                assert rgb_filename == inf_filename

        assert len(self.rgb) == len(self.inf)
        self.transform = transform

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        rgb_image = Image.open(os.path.join(self.rgb[index])).convert('RGB')
        inf_image = Image.open(os.path.join(self.inf[index])).convert('RGB')
        rgb_image, inf_image = self.transform(rgb_image, inf_image)
        return rgb_image, inf_image
    
    