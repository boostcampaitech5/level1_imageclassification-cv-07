import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from common.dataset import MaskBaseDataset, Subset_transform
from common.augmentation import BaseAugmentation, CustomAugmentation
from common.loss import create_criterion

from architecture.model import BaseModel
from typing import Union

from common.pytorchtools import EarlyStopping
import wandb
from sklearn.metrics import f1_score

from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

def seed_everything(seed: int):
    """실험의 재현가능성을 위해 seed를 설정하는 함수.

    Args:
        seed (int): 사용자가 정의한 정수 값.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    """validation data에 대한 모델의 예측 결과를 시각화하는 함수.

    Args:
        np_images (_type_): 입력 이미지(주의: tensor와 ndarray는 차원 나열 방식이 다름)
        gts (_type_): 정답(=ground truth, label)
        preds (_type_): 모델 예측 결과
        n (int, optional): 몇 개의 이미지를 나타낼 것인지 결정하는 파라미터. Defaults to 16.
        shuffle (bool, optional): 입력된 batch 단위의 이미지 중에서, 무작위로 선택할 지를 결정하는 파라미터. Defaults to False.

    Returns:
        _type_: 예측 결과가 시각화된 figure
    """
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path: Union[str, Path], exist_ok=False) -> str:
    """ 경로명을 자동으로 증가시켜주는 함수.
    e.g. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str): f"{model_dir}/{args.name}".
        exist_ok (bool): 경로명을 증가시킬지 선택. Defaults to False.

    Returns:
        str: 증가시킨 경로명
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train_with_cutmix(data_dir: str, model_dir: str, args: argparse.Namespace):
    
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # --dataset
    dataset_module = getattr(import_module('common.dataset'), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes

    # --augmentation
    transform_module = getattr(import_module('common.augmentation'), args.augmentation)
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std
    )

    # -- dataloader
    train_set, val_set = dataset.split_dataset() # 우선 train/val을 분리

    # -- CutMix
    train_set = Subset_transform(train_set, transform=transform) # resize, normalize를 적용
    train_set = CutMix(train_set, num_class=num_classes, beta=1.0, prob=0.5, num_mix=2) # train에만 적용
    
        
    val_set = Subset_transform(val_set, transform=A.Compose([
            A.Resize(height=args.resize[0], width=args.resize[1]),
            A.Normalize(dataset.mean, dataset.std),
            ToTensorV2()
        ]))
    
    train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("architecture.model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = CutMixCrossEntropyLoss(True)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train_loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss} || lr {current_lr}"
                )
                wandb.log({
                    "Train/loss": train_loss
                    })
            
                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)

                loss_item = criterion(outs, labels).item()
                val_loss_items.append(loss_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            if val_loss < best_val_loss:
                print(f"New best model for val loss : {val_loss}! saving the best model..")
                torch.save(model.module.state_dict(), "best.pth")
                best_val_loss = val_loss
            torch.save(model.module.state_dict(), "last.pth")
            print(
                f"loss: {val_loss} || best loss: {best_val_loss}"
            )
            wandb.log({
                "Val/loss": val_loss,
            })
            print()


def train_with_fold(data_dir: str, model_dir: str, args: argparse.Namespace, num_folds: int =5):
    """k-fold, stratified-fold, grouped-fold 방식으로 모델을 학습시킬 때 사용하는 함수

    Args:
        data_dir (str): 입력 데이터 경로
        model_dir (str): 학습한 모델을 저장할 경로
        args (argparse.Namespace): 사용자가 입력한 arguments
        num_folds (int, optional): 폴드의 개수. Defaults to 5.
    """

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("common.dataset"), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes # 18

    # -- augmentation
    transform_module = getattr(import_module("common.augmentation"), args.augmentation)
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- create fold (mode에 따라 다른 방식으로 fold를 생성)
    fold = None
    if args.mode == "k": # k-fold 방식으로 dataloader를 생성
        fold = KFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
        splitted_fold = fold.split(dataset)
    elif args.mode == "s": # stratified-fold 방식으로 dataloader를 생성
        fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
        gt = []
        for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels):
            gt.append(dataset.encode_multi_class(mask, gender, age))
        
        splitted_fold = fold.split(dataset, gt)
    elif args.mode == "g": # grouped-fold 방식으로 dataloader를 생성 (아직 사용해보지 않았음)
        fold = GroupKFold(n_splits=num_folds)
    else:
        print(f"Unknown inputs: you must select mode in k(k-fold), s(stratified-fold), g(grouped-fold)")
        exit()

    
    fold_model_dir = increment_path(os.path.join(model_dir, 'fold_model'))
    for fold, (train_ids, val_ids) in enumerate(splitted_fold):

        seed_everything(args.seed)

        save_dir = increment_path(os.path.join(fold_model_dir, args.name))

        # Print
        print(f'Fold {fold}')
        print('-'*20)

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        # 현재 fold에서 train/val dataloader를 정의
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            sampler=train_subsampler,
            pin_memory=use_cuda,
            drop_last=True,   
        )

        val_loader = DataLoader(
            dataset,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            sampler=val_subsampler,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        model_module = getattr(import_module("architecture.model"), args.model) # default: BaseModel
        model = model_module(
            num_classes=num_classes,
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion) # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)


        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                # val_acc = np.sum(val_acc_items) / len(val_set)
                val_acc = np.sum(val_acc_items) / int(len(dataset) * (1 / num_folds))
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()


def train(data_dir: str, model_dir: str, args: argparse.Namespace):
    """모델 학습에 사용하는 함수

    Args:
        data_dir (str): 학습에 사용될 데이터의 경로
        model_dir (str): 학습된 모델의 pth 파일 및 log들을 저장하는 경로
        args (argparse.Namespace): 사용자가 입력한 argument 값들
    """
    
    seed_everything(args.seed)

    early_stopping = EarlyStopping(patience = args.early_stopping, verbose = True)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("common.dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("common.augmentation"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_set = Subset_transform(train_set, transform=transform)

    # -- valid_transform
    if args.valid_transform:
        val_set = Subset_transform(val_set, transform=transform)
    else:
        val_set = Subset_transform(val_set, transform=A.Compose([
            A.Resize(height=args.resize[0], width=args.resize[1]),
            A.Normalize(dataset.mean, dataset.std),
            A.pytorch.ToTensorV2()
        ]))

    # -- imbalanced
    if args.imbalanced:
        train_loader = DataLoader(
            train_set,
            sampler = ImbalancedDatasetSampler(train_set),
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("architecture.model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            # 각 iter마다 균등한 labels를 통해 학습하는지 확인용 -> 안돼면 말해주세요!!
            # print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # 후에 fine-tuning을 시도한다면 optimizer가 여러 개 필요할 듯 해서, train.py를 독립적으로 만들어야할 수도 있음.

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                wandb.log({
                    "Train/loss": train_loss,
                    "Train/accuracy": train_acc,
                    "Train F1 Score": f1_score(labels.clone().detach().cpu().numpy(), preds.clone().detach().cpu().numpy(), average='macro')
                })

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            wandb.log({
                "Val/loss": val_loss,
                "Val/accuracy": val_acc,
                "Val F1 score": f1_score(labels.clone().detach().cpu().numpy(), preds.clone().detach().cpu().numpy(), average='macro')
            })
            print()
        
        if args.early_stopping > 0:
            early_stopping(val_loss, model)
            if early_stopping.early_stop: 
                break


if __name__ == '__main__':
    wandb.init(project="image-classification-competitions", reinit=True)
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--mode', type=str, default='plain', help='whether using fold training schemes (default: plain)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--early_stopping', type=int, default=0, help="training stops when the loss increases n times in a row")
    parser.add_argument('--imbalanced', type=bool, default=False, help='whether using Imbalanced Dataset Sampling')
    parser.add_argument('--valid_transform', type=bool, default=False, help='whether applying transform to validation dataset')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    wandb.config.update(args)
    wandb.run.name = args.name
    data_dir = args.data_dir
    model_dir = args.model_dir
    mode = args.mode

    if mode == 'plain':
        train(data_dir, model_dir, args)
    elif mode in ['k', 's', 'g']: # k-fold, stratified-fold, grouped-fold
        train_with_fold(data_dir, model_dir, args)
    elif mode == 'cutmix':
        train_with_cutmix(data_dir, model_dir, args)
