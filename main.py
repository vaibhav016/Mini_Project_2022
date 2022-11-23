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
from torchsummary import summary

from dataset import build_data_loader
from model import ResNet, BasicBlock
from utils import initialise_optimisers, evaluate, initialise_schedular, initialise_configs, train_epoch

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


import argparse


def train(config, device):
    model = ResNet(
        BasicBlock,
        num_blocks=config.data_config["resnet_blocks"],
        input_channel=config.data_config["input_channel"],
        channels=config.data_config["block_channel"],
    )
    # print("the number of parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (3, 32, 32))
    train_dataloader, validation_dataloader, test_dataloader = build_data_loader(config)
    optimizer = initialise_optimisers(config, model)
    scheduler = initialise_schedular(config, optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    tolerance = config.train_config["tolerance"]
    tolerance_start_counter = 0

    model = model.to(device)
    criterion = criterion.to(device)

    validation_loss_min = torch.inf
    model_path = config.data_config["data_path"] + "model.pt"
    epochs = config.train_config["epochs"]
    print("---------Training Starts-----------")
    for i in range(epochs):
        training_loss, training_accuracy = train_epoch(
            model, train_dataloader, optimizer, criterion, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_dataloader, criterion, device
        )
        scheduler.step(validation_loss)
        print(
            "epoch: %d | train_loss: %.2f| train_accuracy: %.2f | valid_loss: %.2f | valid_accuracy: %.2f"
            % (
                i,
                training_loss,
                training_accuracy,
                validation_loss,
                validation_accuracy,
            )
        )

        if validation_loss < validation_loss_min:
            print("Saving model ...")
            validation_loss_min = validation_loss
            torch.save(
                {
                    "epoch": i + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                },
                model_path,
            )
            tolerance_start_counter = 0
        else:
            # Implementing early stopping
            tolerance_start_counter += 1
            if tolerance_start_counter == tolerance:
                print("Early Stopping at epoch ", i + 1)
                break

    testing_loss, testing_accuracy = evaluate(model, test_dataloader, criterion, device)
    print(
        "testing loss: %.2f | testing accuracy: %.2f" % (testing_loss, testing_accuracy)
    )


def infer(config, device):
    model = ResNet(
        BasicBlock,
        num_blocks=config.data_config["resnet_blocks"],
        input_channel=config.data_config["input_channel"],
        channels=config.data_config["block_channel"],
    )

    # print("the number of parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (3, 32, 32))
    checkpoint = torch.load(config.data_config["saved_model"],map_location=device)
    optimizer = initialise_optimisers(config, model)
    criterion = torch.nn.CrossEntropyLoss()
    # Loading the checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    _, _,test_dataloader = build_data_loader(config)

    testing_loss, testing_accuracy = evaluate(model, test_dataloader, criterion, device)
    print(
        "testing loss: %.4f | testing accuracy: %.4f" % (testing_loss, testing_accuracy)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Deep Learning MiniProject")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_YAML,
        help="The file path of model configuration file",
    )
    parser.add_argument('-tr', '--training', type=int, default=0,  help='Enter 1 for training and 0 for inference')
    parser.add_argument('-d', '--data_path', type=str, help = 'Enter path where you want to save model')
    parser.add_argument('-s', '--saved_model', type=str, help='Enter path of saved model')

    args = parser.parse_args()
    config = initialise_configs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.training == 1:
        train(config, device)
    else:
        infer(config, device)
