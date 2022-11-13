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
from torch.optim.lr_scheduler import MultiplicativeLR, ReduceLROnPlateau

from dataset import build_data_loader
from model import ResNet, BasicBlock, Bottleneck
from utils import initialise_optimisers, train, evaluate
from torchsummary import summary

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

from config import Config
import argparse


def run(config, device):
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
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    tolerance = 10
    tolerance_start_counter = 0

    model = model.to(device)
    criterion = criterion.to(device)

    validation_loss_min = torch.inf
    model_path = config.data_config["data_path"] + "model.pt"
    epochs = config.train_config["epochs"]
    print("---------Training Starts-----------")
    for i in range(epochs):
        training_loss, training_accuracy = train(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Deep Learning MiniProject")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_YAML,
        help="The file path of model configuration file",
    )
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(config, device)
