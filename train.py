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

import os
import torch
from torch import optim

from dataset import build_data_loader
from model import ResNet, BasicBlock, Bottleneck
from utils import initialise_optimisers, train, evaluate

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

from config import Config
import argparse


def run(config, device):
    model = ResNet(BasicBlock,
                    num_blocks=config.data_config["resnet_blocks"],
                    input_planes=config.data_config["input_planes"],
                    layers=config.data_config["block_layers"])

    print("the number of parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_dataloader, validation_dataloader, test_dataloader = build_data_loader(config)
    optimizer = initialise_optimisers(config, model)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    validation_loss_min = torch.inf
    model_path = config.data_config["data_path"] + "model.pt"
    epochs = config.train_config["epochs"]

    # Fill training code here
    for i in range(epochs):
        training_loss, training_accuracy = train(model, train_dataloader, optimizer, criterion, device)
        validation_loss, validation_accuracy = evaluate(model, validation_dataloader, criterion, device)
        print('epoch: %d | train_loss: %.2f| train_accuracy: %.2f | valid_loss: %.2f | valid_accuracy: %.2f' % (
            i, training_loss, training_accuracy, validation_loss, validation_accuracy))
        if validation_loss < validation_loss_min:
            print('Saving model ...')
            validation_loss_min = validation_loss
            testing_loss, testing_accuracy = evaluate(model, test_dataloader, criterion, device)
            print("testing loss: %.2f | testing accuracy: %.2f" % (testing_loss, testing_accuracy))
            # torch.save({
            #         'epoch': i,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': validation_loss,
            #         }, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Deep Learning MiniProject")

    parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
    args = parser.parse_args()
    config = Config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(config, device)
