#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime

import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from captum.attr import LayerGradCam, LayerAttribution
from medmnist import INFO
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
from torch.optim import adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

print(os.path.join(sys.path[0], ".."))
# sys.path.append(os.path.join(sys.path[0], ".."))
from sibnet import sibnet_losses


def main(sibnetloss_on: bool, cd_loss_weight: float, sc_loss_weight: float, num_epochs: int, batch_size: int,
         learning_rate: float, logdir: str):
    # set seeds
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_flag = 'pneumoniamnist'
    resampled_size = (224, 224)
    download = True

    # use z-score normalization for this test
    means = [0.5]
    stds = [0.5]

    NUM_EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    lr = learning_rate
    val_interval = 1

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    logdir_datetime = logdir + "_" + info['python_class'] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=logdir_datetime)

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resampled_size),
        transforms.Normalize(mean=means, std=stds)
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)

    print(train_dataset)
    print("===================")

    # adapt model to set the number of input channels and output classes
    model = resnet18().float()
    model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, n_classes)
    model.to(device)

    optimizer = adam.Adam(params=model.parameters(), lr=lr, betas=(0.93, 0.999))

    for i in np.arange(0, NUM_EPOCHS):
        print('Epoch ' + str(i))

        ce_loss_epoch_train = 0
        cd_loss_epoch_train = 0
        loss_epoch_train = 0
        sc_loss_epoch_train = 0

        targets_epoch = torch.tensor([])
        outputs_epoch = torch.tensor([])

        model.train()

        for inputs, targets in train_loader:
            targets_epoch = torch.cat((targets_epoch, torch.argmax(targets, 1)), 0)

            # get current class weights for the wCE loss
            data_hist = np.zeros(n_classes)
            for elem in targets:
                data_hist[elem] += 1

            data_hist /= BATCH_SIZE

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            ce_weights = torch.Tensor(1 - data_hist).to(device)
            wCEcriterion = nn.CrossEntropyLoss(weight=ce_weights)

            loss = wCEcriterion(outputs, targets.squeeze().long())
            ce_loss_epoch_train += loss.item()

            if sibnetloss_on:
                cam = LayerGradCam(model, layer=model.layer4)
                attr_classes = [torch.Tensor(cam.attribute(inputs, [i] * inputs.shape[0])) for i in range(n_classes)]

                cdcriterion = sibnet_losses.ClassDistinctivenessLoss(device=device)
                cdloss = cdcriterion(attr_classes)

                # or the SC loss, use the saliency map at the input, upsampled to match the input size
                upsampled_attr = [LayerAttribution.interpolate(attr, torch.squeeze(inputs[0]).shape) for attr in
                                  attr_classes]

                sccriterion = sibnet_losses.SimilarityCoherence(device=device, kernel_size=9)
                scloss = sccriterion(upsampled_attr, device)

                loss += cd_loss_weight * cdloss + sc_loss_weight * scloss

                cd_loss_epoch_train += cdloss.item()
                sc_loss_epoch_train += scloss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_epoch_train += loss.item()

            outputs_epoch = torch.cat((outputs_epoch, torch.argmax(outputs, 1).cpu().detach()), 0)

        if sibnetloss_on:
            writer.add_scalar('CD Loss/train', cd_loss_epoch_train, i)
            writer.add_scalar('SC Loss/train', sc_loss_epoch_train, i)

        writer.add_scalar('wCE Loss/train', ce_loss_epoch_train, i)
        writer.add_scalar('Loss/train', loss_epoch_train, i)
        writer.add_scalar('Accuracy/train', accuracy_score(targets_epoch, outputs_epoch), i)
        writer.add_scalar('Balanced accuracy/train', balanced_accuracy_score(targets_epoch, outputs_epoch), i)
        writer.add_scalar('F1/train', f1_score(targets_epoch, outputs_epoch, average='weighted'), i)

        # if we reach a validation interval, evaluate
        if i % val_interval == 0:
            print("Validating...")
            model.eval()

            ce_loss_epoch_val = 0
            cd_loss_epoch_val = 0
            sc_loss_epoch_val = 0
            loss_epoch_val = 0

            targets_epoch_val = torch.tensor([])
            outputs_epoch_val = torch.tensor([])

            for inputs_val, targets_val in val_loader:
                targets_epoch_val = torch.cat((targets_epoch_val, torch.argmax(targets_val, 1)), 0)

                # get current class weights for the wCE loss
                data_hist_val = np.zeros(n_classes)
                for elem in targets_val:
                    data_hist_val[elem] += 1

                data_hist_val /= BATCH_SIZE

                inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                outputs_val = model(inputs_val)
                outputs_val = outputs_val.softmax(dim=-1)

                targets_val = targets_val.float().resize_(len(targets_val), 1)

                ce_weights_val = torch.Tensor(1 - data_hist_val).to(device)

                # weighted cross entropy
                wCEcriterion = nn.CrossEntropyLoss(weight=ce_weights_val)
                loss_val = wCEcriterion(outputs_val, targets_val.squeeze().long())
                ce_loss_epoch_val += loss_val.item()

                if sibnetloss_on:
                    cam = LayerGradCam(model, layer=model.layer4)
                    attr_classes_val = [torch.Tensor(cam.attribute(inputs_val, [i] * inputs_val.shape[0])) for i in
                                        range(n_classes)]

                    # class distinctiveness
                    cdcriterion = sibnet_losses.ClassDistinctivenessLoss(device=device)
                    cdloss_val = cdcriterion(attr_classes_val)

                    # spatial coherence
                    upsampled_attr_val = [LayerAttribution.interpolate(attr, torch.squeeze(inputs_val[0]).shape) for
                                          attr in
                                          attr_classes_val]
                    sccriterion = sibnet_losses.SimilarityCoherence(device=device, kernel_size=9)
                    scloss_val = sccriterion(upsampled_attr_val, device)

                    loss_val += cd_loss_weight * cdloss_val + sc_loss_weight * scloss_val

                    cd_loss_epoch_val += cdloss_val.item()
                    sc_loss_epoch_val += scloss_val.item()

                loss_epoch_val += loss_val.item()
                outputs_epoch_val = torch.cat((outputs_epoch_val, torch.argmax(outputs_val, 1).cpu().detach()), 0)

            if sibnetloss_on:
                writer.add_scalar('CD Loss/val', cd_loss_epoch_val, i)
                writer.add_scalar('SC Loss/val', sc_loss_epoch_val, i)

            writer.add_scalar('wCE Loss/val', ce_loss_epoch_val, i)
            writer.add_scalar('Loss/val', loss_epoch_val, i)
            writer.add_scalar('Accuracy/val', accuracy_score(targets_epoch_val, outputs_epoch_val), i)
            writer.add_scalar('Balanced accuracy/val', balanced_accuracy_score(targets_epoch_val, outputs_epoch_val), i)
            writer.add_scalar('F1/val', f1_score(targets_epoch_val, outputs_epoch_val, average='weighted'), i)

    writer.close()

    # evaluate the model on the test set
    print("Testing")
    model.eval()

    targets_epoch_test = torch.tensor([])
    outputs_epoch_test = torch.tensor([])

    for inputs_test, targets_test in test_loader:
        targets_epoch_test = torch.cat((targets_epoch_test, torch.argmax(targets_test, 1)), 0)

        inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
        outputs_test = model(inputs_test)
        outputs_test = outputs_test.softmax(dim=-1)

        outputs_epoch_test = torch.cat((outputs_epoch_test, torch.argmax(outputs_test, 1).cpu().detach()), 0)

    report = pd.DataFrame(classification_report(targets_epoch_test, outputs_epoch_test,
                                                target_names=info['label'].values(), output_dict=True)).transpose()
    print(report)
    report.to_csv(os.path.join(logdir_datetime, "test_eval.csv"))


if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Run a classification example with or without the sibnet losses')

    parser.add_argument(
        '--sibnetloss_on',
        type=bool,
        default=True,
        help='Path to the current patients top directory.'
    )

    parser.add_argument(
        '--cd_loss_weight',
        type=float,
        default=1.2,
        help='weight of the class distinctiveness loss'
    )

    parser.add_argument(
        '--sc_loss_weight',
        type=float,
        default=0.9,
        help='weight of the spacial coherence loss'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of epochs to train the model'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=25,
        help='Number samples in a batch'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate for the optimizer'
    )

    parser.add_argument(
        '--logdir',
        type=str,
        default="cdloss",
        help='Path where the log files should be saved'
    )

    args = parser.parse_args()

main(args.sibnetloss_on, args.cd_loss_weight, args.sc_loss_weight, args.num_epochs, args.batch_size, args.learning_rate,
     args.logdir)
