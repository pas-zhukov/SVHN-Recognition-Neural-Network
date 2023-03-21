import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_loaders(batch_size, data_train, validation_split=.2):
    # Количество тренировочных примеров
    data_size = data_train.data.shape[0]

    # Определяем количество примеров в фолде валидации
    split = int(np.floor(validation_split * data_size))

    # Список индексов для тренировочных примеров
    indices = list(range(data_size))

    # Рандомизируем положение индексов в списке
    np.random.shuffle(indices)

    # Определяем списки с индексами примеров для тренировки и для валидации
    train_indices, val_indices = indices[split:], indices[:split]

    # Создаем семплеры, которые будут случайно извлекать данные из набора данных
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Создаем объекты типа ДатаЛоадер, которые будут передавать батчами данные в модель
    train_loader = DataLoader(data_train, batch_size=batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(data_train, batch_size=batch_size,
                            sampler=val_sampler)

    return train_loader, val_loader


def get_SVHN():
    data_train = datasets.SVHN('./data/', split='train',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.43, 0.44, 0.47],
                                                        std=[0.20, 0.20, 0.20])
                               ]), download=True
                               )
    data_test = datasets.SVHN('./data/', split='test',
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.43, 0.44, 0.47],
                                                       std=[0.20, 0.20, 0.20])
                              ]), download=True)

    return data_train, data_test