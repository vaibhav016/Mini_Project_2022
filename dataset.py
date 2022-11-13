# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Taken from prev assignment
# train_transforms = transforms.Compose(
#         [
#             transforms.RandomRotation(5),
#             transforms.RandomCrop((32,32), padding=4),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
#         ])

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


def build_data_loader(config):
    path = config.data_config["data_path"]
    batch_size = config.data_config["batch_size"]
    num_workers = config.data_config["num_workers"]

    train_transforms = [
        transforms.RandomRotation(config.data_config["random_rotation"])
    ]
    if config.data_config["crop"]:
        train_transforms.append(transforms.RandomCrop((32, 32), padding=4))
    if config.data_config["horizontal_flip"]:
        train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    train_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    train_transforms = transforms.Compose(train_transforms)

    print("---------Download Cifar10--------")
    train_data = torchvision.datasets.CIFAR10(
        root=path, train=True, transform=train_transforms, download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=path, train=False, transform=test_transforms, download=True
    )
    print("------Download and verification complete-------")

    n_train_examples = int(len(train_data) * config.data_config["validation_split"])
    n_valid_examples = len(train_data) - n_train_examples
    train_data, valid_data = data.random_split(
        train_data, [n_train_examples, n_valid_examples]
    )

    valid_data.dataset.transform = test_transforms
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_dataloader = data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("Validation split: ", config.data_config["validation_split"])
    print("Number of Training examples ", n_train_examples)
    print("Number of Validation examples ", n_valid_examples)

    print("-------- Train and Test DataLoader building finished--------")

    return train_dataloader, valid_dataloader, test_dataloader
