import datetime
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import dataloader
import transforms as T
import utils
import wandb
from sklearn.metrics import classification_report
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


try:
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc = utils.accuracy(output, target)
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc'].update(acc[0].item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    return metric_logger.loss.value, metric_logger.acc.value


def evaluate(model, criterion, data_loader, visualize, epoch, device, name, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    idx = 1
    target_class = torch.Tensor([])
    pred_class = torch.Tensor([])

    if visualize:
        target_layers = [model.layer4[-1]]
        with GradCAM(model=model, target_layers=target_layers) as cam_extractor:
            for image, target in metric_logger.log_every(data_loader, print_freq, header):

                im = image

                target_class = torch.cat((target_class, target), dim=0)

                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                _, pred = output.topk(1, 1, True, True)
                loss = criterion(output, target)

                acc = utils.accuracy(output, target)

                pred_class = torch.cat((pred_class, pred.cpu()), dim=0)

                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc'].update(acc[0].item(), n=batch_size)
                    # Resize the CAM and overlay it
                for i in range(len(im)):
                    # only do visual on predicted malaria cells
                    if pred.t()[0][i].item() != 0:
                        orig_img = np.moveaxis(im[i].numpy(), 0, -1) * .2 + .5
                        activation_map = cam_extractor(input_tensor=image, aug_smooth=True, eigen_smooth=True)
                        grayscale_cam = activation_map[i, :]
                        visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
                        # activation_map = cam_extractor(input_tensor=output, target_category=target, aug_smooth=True,
                        #                                eigen_smooth=True)
                        # result = overlay_mask(to_pil_image(orig_img), to_pil_image(activation_map[0].squeeze(0), mode='F'),
                        #                       alpha=0.5)
                        fig, ax = plt.subplots()
                        ax.imshow(visualization, vmin=0, vmax=1)
                        ax.set_title('Cell' + str(idx))
                        ax.tick_params(axis='both', labelsize=0, length=0)
                        ax.set(xlabel=("Predicted Label: " + str(pred.t()[0][i].item()) + ' Actual: ' + str(
                            target.cpu()[i].item())))
                        fig.savefig(os.path.join('save_models', name, 'visualize', str(idx) + '.png'))
                        plt.close('all')
                        idx += 1
                # free memory
                del image
                del target
                del output
    else:
        with torch.no_grad():
            for image, target in metric_logger.log_every(data_loader, print_freq, header):

                target_class = torch.cat((target_class, target), dim=0)

                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                _, pred = output.topk(1, 1, True, True)
                loss = criterion(output, target)

                acc = utils.accuracy(output, target)

                pred_class = torch.cat((pred_class, pred.cpu()), dim=0)

                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc'].update(acc[0].item(), n=batch_size)

                # free memory
                del image
                del target
                del output
    target_names = ['red_blood_cell', 'gametocyte', 'ring', 'schizont', 'trophozoite']
    results = classification_report(target_class, pred_class, target_names=target_names)
    report = classification_report(target_class, pred_class, target_names=target_names, output_dict=True)
    cm = confusion_matrix(target_class, pred_class, normalize='pred')
    df_cm = pd.DataFrame(cm, index=[i for i in target_names],
                         columns=[i for i in target_names])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="OrRd")
    # disp = ConfusionMatrixDisplay(cm, display_labels=target_names).plot()
    print(results)
    os.makedirs(os.path.join('save_models', name), exist_ok=True)
    plt.savefig(os.path.join('save_models', name, f'epoch{epoch}_confmat.png'))
    plt.close('all')
    with open(os.path.join('save_models', name, f'epoch{epoch}.txt'), 'w') as file:
        file.write(results)

    # free classification report var
    del target_class
    del pred_class

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc {top1.global_avg:.3f}'
          .format(top1=metric_logger.acc))

    return metric_logger.loss.value, report['macro avg']['f1-score']


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, testdir, args):
    # Data loading code
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)

    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        # auto_augment_policy = getattr(args, "auto_augment", None)
        # random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = dataloader.CellClassificationDataset(
            get_transform(True, base_size=resize_size, crop_size=crop_size),
            'train')

        if args.cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    # print("Took", time.time() - st)

    # loading test data
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = dataloader.CellClassificationDataset(
            get_transform(False, base_size=resize_size, crop_size=crop_size),
            'val')
        if args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

        # loading testonly data
        cache_path = _get_cache_path(testdir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print("Loading dataset_testonly from {}".format(cache_path))
            dataset_testonly, _ = torch.load(cache_path)
        else:
            dataset_testonly = dataloader.CellClassificationDataset(
                get_transform(False, base_size=resize_size, crop_size=crop_size),
                'test')
            if args.cache_dataset:
                print("Saving dataset_testonly to {}".format(cache_path))
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset_testonly, testdir), cache_path)

    # creating data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        testonly_sampler = torch.utils.data.distributed.DistributedSampler(dataset_testonly)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        testonly_sampler = torch.utils.data.SequentialSampler(dataset_testonly)

    return dataset, dataset_test, train_sampler, test_sampler, dataset_testonly, testonly_sampler


# takes in boolean for train vs. test transforms
def get_transform(train, base_size, crop_size):
    min_size = int((0.6 if train else 1.0) * base_size)
    max_size = int((1.5 if train else 1.0) * base_size)
    transforms = [T.RandomResize(min_size, max_size)]
    if train:
        transforms.append(T.RandomCrop(crop_size))
        # transforms.append(torchvision.transforms.RandomResizedCrop(250, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333)))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomRotate90(0.5))
        transforms.append(torchvision.transforms.ColorJitter())
        transforms.append(T.GaussianBlur((7, 7), 2))
    else:
        transforms.append(T.CenterCrop(crop_size))
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(torchvision.transforms.RandomErasing())
    transforms.append(T.Normalize(mean=.5, std=.2))

    return T.Compose(transforms)


def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.name:
        utils.mkdir(os.path.join('save_models', args.name))
        if args.visualize:
            utils.mkdir(os.path.join('save_models', args.name, 'visualize'))

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    test_dir = os.path.join(args.data_path, 'test')
    dataset, dataset_test, train_sampler, test_sampler, dataset_testonly, testonly_sampler = load_data(train_dir, val_dir, test_dir, args)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_testonly = torch.utils.data.DataLoader(
        dataset_testonly, batch_size=args.batch_size,
        sampler=testonly_sampler, num_workers=args.workers, pin_memory=True)

    # creating model
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

    if args.pretrained & (args.model == 'resnet50'):
        count = 0
        for child in model.children():
            count += 1
            if count == 7:
                break
            for param in child.parameters():
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # weight = torch.tensor([2.02791630e-01, 2.94467433e+01, 4.68634146e+01, 2.84651852e+02, 9.98129870e+01]).to(device)
    weight = torch.tensor([0.59363636, 0.61314554, 0.98195489, 5.93636364, 2.00923077]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    # criterion = focalloss.FocalLoss(weight=weight)
    # criterion = torch.hub.load('adeelh/pytorch-multi-class-focal-loss',
    #                            model='focal_loss',
    #                            alpha=torch.tensor([2.02791630e-01, 2.94467433e+01, 4.68634146e+01, 2.84651852e+02,
    #                                                9.98129870e+01]),
    #                            gamma=2,
    #                            reduction='mean',
    #                            device='cuda',
    #                            dtype=torch.float32,
    #                            force_reload=False)

    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        epoch = args.start_epoch

    if args.test_only:
        evaluate(model, criterion, data_loader_testonly, False, 9999, device=device, name=args.name)
        return

    if args.wandb:
        # weights and bias tracking setup
        wandb.config = {
            "experiment_name": args.name,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.opt,
            # "lr_scheduler": f'step lr- step size: {args.lr_step_size}, gamma: {args.lr_gamma}',
            "lr_scheduler": 'cosine_annealing',
            "pretrained": args.pretrained,
            "loss function": "focal loss"
        }

        # for weights and bias tracking
        wandb.init(config=wandb.config, project="cap5516finalproject", entity="joefioresi718", name=args.name)

    # start training
    start_time = time.time()
    best_loss = 5
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, acc_train = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, args.apex)
        lr_scheduler.step()
        loss_val, f1_val = evaluate(model, criterion, data_loader_test, False, epoch, device=device, name=args.name)

        if args.wandb:
            # log loss and accuracies
            wandb.log({
                "val_loss": loss_val,
                "train_loss": loss_train,
                "val_f1": f1_val,
                "train_acc": acc_train
            })
            # Optional
            # wandb.watch(model)

        if args.name:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            # utils.save_on_master(
            #     checkpoint,
            #     os.path.join('save_models/' + args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join('save_models', args.name, 'checkpoint.pth'))
            if loss_val < best_loss:
                best_loss = loss_val
                utils.save_on_master(checkpoint, os.path.join('save_models', args.name, 'best_model.pth'))

    if args.visualize:
        checkpoint = torch.load(os.path.join('save_models', args.name, 'best_model.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=not args.test_only)
        test_loss, test_f1 = evaluate(model, criterion, data_loader_testonly, False, epoch+1, device=device, name=args.name)
        if args.wandb:
            # log loss and accuracies
            wandb.log({
                "test_loss": test_loss,
                "test_f1": test_f1
            })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument('--wandb', default=True, help='weights and bias')
    parser.add_argument('--name', default='reports', help='filesystem name')
    parser.add_argument('--data-path', default='data/', help='dataset')
    parser.add_argument('--anno-dir', default='files/')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=20, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='Test/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--visualize', default=True, action='store_true', help='visualize output')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        default=False,
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=True,
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    # parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
