from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import torch


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class flowerDataLoader(BaseDataLoader):
    """
    flowerData data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir
        data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转
                                         transforms.CenterCrop(224),  # 中心裁剪
                                         transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转 p概率=0.5
                                         transforms.RandomVerticalFlip(p=0.5),  # 竖直翻转
                                         transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                         transforms.RandomGrayscale(p=0.025),  # 转灰度图
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         # 标准化，均值，标准差
                                         ]),
            'valid': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ]),
        }
        """
        制作好数据源：
        - data_transforms中指定了所有图像预处理操作
        - ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
        """
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms['train']) for x in ['flower_data']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['flower_data']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['flower_data']}
        class_names = image_datasets['flower_data'].classes

        # dataloaders.dataset.targets=cat_to_name
        self.dataset=dataloaders['flower_data'].dataset

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class catdogDataLoader(BaseDataLoader):
    """
    flowerData data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir
        data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转
                                         transforms.CenterCrop(224),  # 中心裁剪
                                         transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转 p概率=0.5
                                         transforms.RandomVerticalFlip(p=0.5),  # 竖直翻转
                                         transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                         transforms.RandomGrayscale(p=0.025),  # 转灰度图
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         # 标准化，均值，标准差
                                         ]),
            'valid': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ]),
        }
        """
        制作好数据源：
        - data_transforms中指定了所有图像预处理操作
        - ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
        """
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms['train']) for x in ['cat_dog_data']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['cat_dog_data']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['cat_dog_data']}
        class_names = image_datasets['cat_dog_data'].classes

        # dataloaders.dataset.targets=cat_to_name
        self.dataset=dataloaders['cat_dog_data'].dataset

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)