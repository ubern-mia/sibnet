import os
import argparse
import time
from tqdm import trange
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from tensorboardX import SummaryWriter
from collections import OrderedDict
from models import ResNet18, ResNet50
from captum.attr import LayerGradCam, LayerAttribution

import medmnist
from medmnist import INFO
from sibnet import evaluator
from sibnet import sibnet_losses


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, resize, as_rgb, run, cdloss,
         cdloss_weight, scloss, scloss_weigh):
    lr = 0.001
    gamma = 0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=2*batch_size,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=2*batch_size,
                                  shuffle=False)

    print('==> Building and training model...')

    model = resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels,
                                                                                      num_classes=n_classes)
    model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.to(device)

    train_evaluator = evaluator.Evaluator(data_flag, 'train')
    val_evaluator = evaluator.Evaluator(data_flag, 'val')
    test_evaluator = evaluator.Evaluator(data_flag, 'test')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'wceloss', 'cdloss', 'scloss', 'auc', 'acc', 'balacc']
    train_logs = [log + '/train' for log in logs]
    val_logs = [log + '/val' for log in logs]
    # test_logs = ['test_' + log for log in logs]
    # log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)
    log_dict = OrderedDict.fromkeys(train_logs + val_logs, 0)

    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = model

    global iteration
    global scloss_scale
    scloss_scale = 1
    iteration = 0

    for epoch in trange(num_epochs):

        train_loss = train(model, train_loader, cdloss, scloss, optimizer, device, writer, n_classes)

        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, device, run, n_classes, cdloss,
         cdloss_weight, scloss, scloss_weight)
        val_metrics = test(model, val_evaluator, val_loader, task, device, run, n_classes, cdloss,
         cdloss_weight, scloss, scloss_weight)
        test_metrics = test(model, test_evaluator, test_loader, task, device, run, n_classes, cdloss,
         cdloss_weight, scloss, scloss_weight)

        # scheduler.step()

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)

        cur_auc = val_metrics[-3]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = model
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, device, run, n_classes, cdloss,
         cdloss_weight, scloss, scloss_weight, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, task, device, run, n_classes, cdloss,
         cdloss_weight, scloss, scloss_weight, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, task, device, run, n_classes, cdloss,
         cdloss_weight, scloss, scloss_weight, output_root)

    # [test_loss, test_loss_wce, test_loss_cd, test_loss_sc, auc, acc, balacc]
    train_log = 'train  auc: %.5f  acc: %.5f bal_acc: %.5f\n' % (train_metrics[-3], train_metrics[-2], train_metrics[-1])
    val_log = 'val  auc: %.5f  acc: %.5f bal_acc: %.5f\n' % (val_metrics[-3], val_metrics[-2], train_metrics[-1])
    test_log = 'test  auc: %.5f  acc: %.5f bal_acc: %.5f\n' % (test_metrics[-3], test_metrics[-2], train_metrics[-1])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)

    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)

    writer.close()


def train(model, train_loader, cdloss, scloss, optimizer, device, writer, n_classes):
    total_loss = []
    global iteration
    global scloss_scale

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # get current class weights for the wCE loss
        data_hist = np.zeros(n_classes)
        for elem in targets:
            data_hist[elem] += 1

        data_hist /= train_loader.batch_size

        ce_weights = torch.Tensor(data_hist).to(device)
        criterion = nn.CrossEntropyLoss(weight=ce_weights)

        targets = torch.squeeze(targets, 1).long().to(device)

        loss = criterion(outputs, targets)

        if cdloss or scloss:
            cam = LayerGradCam(model, layer=model.layer4)
            attr_classes = [torch.Tensor(cam.attribute(inputs, [i] * inputs.shape[0])) for i in range(n_classes)]

            if cdloss:
                cdcriterion = sibnet_losses.ClassDistinctivenessLoss(device=device)
                cdloss_value = cdcriterion(attr_classes)
                loss += cdloss_value

            if scloss:
                upsampled_attr_val = [LayerAttribution.interpolate(attr, torch.squeeze(inputs[0]).shape) for
                                      attr in attr_classes]
                sccriterion = sibnet_losses.SpatialCoherenceConv(device=device, kernel_size=9)
                scloss_value = scloss_scale * sccriterion(upsampled_attr_val, device=device)

                loss += scloss_value

        if iteration == 1 and scloss:
            scloss_scale = 1 / scloss_value.item()

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, device, run, n_classes, cdloss,
         cdloss_weight, scloss, scloss_weight, save_folder=None):
    model.eval()

    total_loss = []
    total_loss_wce = []
    total_loss_cd = []
    total_loss_sc = []

    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            targets = torch.squeeze(targets, 1).long().to(device)

            # get current class weights for the wCE loss
            data_hist = np.zeros(n_classes)
            for elem in targets:
                data_hist[elem] += 1

            data_hist /= data_loader.batch_size

            ce_weights = torch.Tensor(data_hist).to(device)
            criterion = nn.CrossEntropyLoss(weight=ce_weights)

            wceloss = criterion(outputs, targets)
            loss = wceloss

            cdloss_value = torch.Tensor([np.NaN])
            scloss_value = torch.Tensor([np.NaN])

            if cdloss or scloss:
                cam = LayerGradCam(model, layer=model.layer4)
                attr_classes = [torch.Tensor(cam.attribute(inputs, [i] * inputs.shape[0])) for i in range(n_classes)]

                if cdloss:
                    cdcriterion = sibnet_losses.ClassDistinctivenessLoss(device=device)
                    cdloss_value = cdcriterion(attr_classes)
                    loss += cdloss_weight * cdloss_value

                if scloss:
                    upsampled_attr_val = [LayerAttribution.interpolate(attr, torch.squeeze(inputs[0]).shape) for
                                          attr in attr_classes]
                    sccriterion = sibnet_losses.SpatialCoherenceConv(device=device, kernel_size=9)
                    scloss_value = scloss_scale * sccriterion(upsampled_attr_val, device=device)
                    loss += scloss_weight * scloss_value

            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss_wce.append(wceloss.item())
            total_loss_sc.append(scloss_value.item())
            total_loss_cd.append(cdloss_value.item())
            total_loss.append(loss.item())

            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc, balacc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)
        test_loss_wce = sum(total_loss_wce) / len(total_loss_wce)
        test_loss_cd = sum(total_loss_cd) / len(total_loss_cd)
        test_loss_sc = sum(total_loss_sc) / len(total_loss_sc)

        return [test_loss, test_loss_wce, test_loss_cd, test_loss_sc, auc, acc, balacc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='breastmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        default=True,
                        action="store_true")
    parser.add_argument('--resize',
                        default=True,
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        default=False,
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--scloss',
                        default=True,
                        help='Bool to turn the spatial coherence loss on or off',
                        type=bool)
    parser.add_argument('--scloss_weight',
                        default=0.9,
                        help='Weight of the spatial coherence loss.',
                        type=float)
    parser.add_argument('--cdloss',
                        default=False,
                        help='Bool to turn the class distinctiveness loss on or off',
                        type=bool)
    parser.add_argument('--cdloss_weight',
                        default=1.2,
                        help='Weight of the class distinctiveness loss.',
                        type=float)

    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    resize = args.resize
    as_rgb = args.as_rgb
    cdloss = args.cdloss
    cdloss_weight = args.cdloss_weight
    scloss = args.scloss
    scloss_weight = args.scloss_weight
    run = args.run

    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, resize, as_rgb, run, cdloss,
         cdloss_weight, scloss, scloss_weight)
