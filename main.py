import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from data_functions import *
from metrics_functions import *

#Подключим видеокарту!
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)

# Загрузка датасета SVHN
data_train, data_test = get_SVHN()

# Лоадеры для тренировки и валидации
train_loader, val_loader = get_loaders(batch_size=64, data_train=data_train, validation_split=.2)


# Модуль (слой), превращающий картинку 3x32x32 в вектор 1x3072
class Flattener(nn.Module):
    def forward(self, x):
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)


# Определяем структуру нейронной сети. У нас будет 1 скрытый слой из 1000 нейронов
nn_model = nn.Sequential(
    Flattener(),
    nn.Linear(3 * 32 * 32, 1000),
    nn.LeakyReLU(inplace=True),
    nn.BatchNorm1d(1000),
    nn.Linear(1000, 10)
)
nn_model.type(torch.cuda.FloatTensor)
nn_model.to(device)

# Определяем функцию потерь и выбираем оптимизатор
loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
optimizer = optim.Adagrad(nn_model.parameters(), lr=1e-3, weight_decay=1e-1)

# Будем также использовать LR Annealing
'''scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda ep: 0.5, verbose=True)'''

# Лучше будем снижать LR на плато
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1, patience=5)

# Создадим списки для сохранения величины функции потерь, точности на тренировки и валидации - на каждом этапе (эпохе)
loss_history = []
train_history = []
val_history = []



# Запускаем тренировку!
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    nn_model.train()  # Enter train mode

    loss_accum = 0
    correct_samples = 0
    total_samples = 0
    for i_step, (x, y) in enumerate(train_loader):

        # Сохраняем наши тензоры в памяти видеокарты, чтобы всё посчитать побыстрее
        x = x.to(device)
        y = y.to(device)

        # Получаем предсказание с существующими весами
        prediction = nn_model(x)
        # Считаем величину функции потерь
        loss_value = loss(prediction, y)
        # Очищаем градиент
        optimizer.zero_grad()
        # Считаем свежий градиент обратным проходом
        loss_value.backward()
        # Обновляем веса
        optimizer.step()

        # Определяем индексы, соответствующие выбранным моделью лейблам
        _, indices = torch.max(prediction, dim=1)
        # Сравниваем с ground truth, сохраняем количество правильных ответов
        correct_samples += torch.sum(indices == y)
        # Сохраняем количество всех предсказаний
        total_samples += y.shape[0]

        # Аккумулируем значение функции потерь, это пригодится далее
        loss_accum += loss_value

    ave_loss = loss_accum / (i_step + 1)
    train_accuracy = float(correct_samples) / total_samples
    val_accuracy = compute_accuracy(nn_model, val_loader)

    loss_history.append(float(ave_loss))
    train_history.append(train_accuracy)
    val_history.append(val_accuracy)

    scheduler.step(ave_loss)

    print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))


# Проверяем модель на test set
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64)
test_accuracy = compute_accuracy(nn_model, test_loader)
print("Test accuracy: %2.4f" % test_accuracy)

torch.onnx.export(nn_model, data_test, 'svhn.onnx', input_names=["image"], output_names=["Numbers Probabilities"])