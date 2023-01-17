import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

"""
训练自己的语义分割模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为png图片，无需固定大小，传入训练前会自动进行resize。
   由于许多同学的数据集是网络上下载的，标签格式并不符合，需要再度处理。一定要注意！标签的每个像素点的值就是这个像素点所属的种类。
   网上常见的数据集总共对输入图片分两类，背景的像素点值为0，目标的像素点值为255。这样的数据集可以正常运行但是预测是没有效果的！
   需要改成，背景的像素点值为0，目标的像素点值为1。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
"""


def main(args):
    Cuda = True
    distributed = False  # 用于指定是否使用单机多卡分布式运行
    sync_bn = False  # 是否使用sync_bn，DDP模式多卡可用
    fp16 = args.amp  # 混合精度训练
    num_classes = args.num_classes  # num_classes + background
    backbone = "xception"  # 所使用的的主干网络 mobilenet xception
    pretrained = False  # 使用主干网络预训练
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = False，Freeze_Train = False，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = args.model_path
    # ---------------------------------------------------------#
    #   downsample_factor   下采样的倍数8、16
    #                       8下采样的倍数较小、理论上效果更好。
    #                       但也要求更大的显存
    # ---------------------------------------------------------#
    downsample_factor = 8  # 下采样的倍数8、16 8下采样的倍数较小、理论上效果更好 但也要求更大的显存
    #  输入图片的大小
    input_shape = [512, 512]
    # ------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    # ------------------------------------------------------------------#
    Init_Epoch = args.init_epoch
    Freeze_Epoch = args.epochs_freeze
    Freeze_batch_size = args.batch_size_freeze
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = args.epochs_unfreeze
    Unfreeze_batch_size = args.batch_size_unfreeze
    Freeze_Train = args.freeze_train  # 是否进行冻结训练 默认先冻结主干训练后解冻训练

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = args.init_lr
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = args.optimizer
    momentum = args.momentum
    weight_decay = args.weight_decay
    lr_decay_type = "cos"  # 使用到的学习率下降方式，可选的有'step'、'cos'
    save_period = args.save_freq  # 多少个epoch保存一次权值
    save_dir = "./logs"  # 权值与日志文件保存的文件夹

    # ------------------------------------------------------------------#
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP
    #   （二）此处设置评估参数较为保守，目的是加快评估速度
    # ------------------------------------------------------------------#
    eval_flag = True  # 是否在训练时进行评估，评估对象为验证集
    eval_period = 1  # 代表多少个epoch评估一次，不建议频繁的评估
    # 评估需要消耗较多的时间，频繁评估会导致训练非常慢

    VOCdevkit_path = args.data_path  # 数据集路径
    dice_loss = args.dice_loss
    focal_loss = args.focal_loss  # 是否使用focal loss来防止正负样本不平衡【实验观察focal loss的效果】
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)  # 给不同的种类设置不同的损失权值
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = min([os.cpu_count(), Freeze_batch_size, Unfreeze_batch_size, 8])
    ngpus_per_node = torch.cuda.device_count()  # 设置用到的显卡
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(
                f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training..."
            )
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    # 初始化模型参数 载入预训练权重
    model = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        downsample_factor=downsample_factor,
        pretrained=pretrained,
    )
    # 若不载入预训练权重参数 初始化模型的权重参数
    if not pretrained:
        weights_init(model)
    if model_path != "":
        if local_rank == 0:
            print("Load weights {}.".format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print(
                "\nSuccessful Load Key:",
                str(load_key)[:500],
                "……\nSuccessful Load Key Num:",
                len(load_key),
            )
            print(
                "\nFail To Load Key:",
                str(no_load_key)[:500],
                "……\nFail To Load Key num:",
                len(no_load_key),
            )
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
        )
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # 混合精度训练
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    #   多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # 多卡并行训练
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                module=model_train, device_ids=[local_rank], find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #   读取数据集对应的txt
    with open(
        os.path.join(VOCdevkit_path, "SUIM2022/ImageSets/Segmentation/train.txt"), "r"
    ) as f:
        train_lines = f.readlines()
    with open(
        os.path.join(VOCdevkit_path, "SUIM2022/ImageSets/Segmentation/val.txt"), "r"
    ) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes,
            backbone=backbone,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"
                % (optimizer_type, wanted_step)
            )
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"
                % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step)
            )
            print(
                "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"
                % (total_step, wanted_step, wanted_epoch)
            )

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    UnFreeze_flag = False
    # ------------------------------------#
    #   冻结一定部分训练
    # ------------------------------------#
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # -------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    # -------------------------------------------------------------------#
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == "adam" else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
    if backbone == "xception":
        lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
    Init_lr_fit = min(
        max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max
    )  # Init_lr_fit = 7e-3
    Min_lr_fit = min(
        max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
    )  # Min_lr_fit = 7e-5

    #   根据optimizer_type选择优化器
    optimizer = {
        "adam": optim.Adam(
            params=model.parameters(),
            lr=Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        ),
        "sgd": optim.SGD(
            params=model.parameters(),
            lr=Init_lr_fit,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        ),
    }[optimizer_type]

    #   获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type=lr_decay_type,  # "cos"
        lr=Init_lr_fit,
        min_lr=Min_lr_fit,
        total_iters=UnFreeze_Epoch,
    )

    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    train_dataset = DeeplabDataset(
        annotation_lines=train_lines,
        input_shape=input_shape,
        num_classes=num_classes,
        train=True,
        dataset_path=VOCdevkit_path,
    )
    val_dataset = DeeplabDataset(
        annotation_lines=val_lines,
        input_shape=input_shape,
        num_classes=num_classes,
        train=False,
        dataset_path=VOCdevkit_path,
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=train_sampler,
    )

    gen_val = DataLoader(
        dataset=val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=val_sampler,
    )

    # ----------------------#
    #   记录eval的map曲线
    # ----------------------#
    if local_rank == 0:
        eval_callback = EvalCallback(
            net=model,
            input_shape=input_shape,
            num_classes=num_classes,
            image_ids=val_lines,
            dataset_path=VOCdevkit_path,
            log_dir=log_dir,
            cuda=Cuda,
            eval_flag=eval_flag,
            period=eval_period,
        )
    else:
        eval_callback = None

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # ---------------------------------------#
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        # ---------------------------------------#
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 16
            lr_limit_max = 5e-4 if optimizer_type == "adam" else 1e-1
            lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
            if backbone == "xception":
                lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
            Init_lr_fit = min(
                max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max
            )
            Min_lr_fit = min(
                max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2),
                lr_limit_max * 1e-2,
            )
            # ---------------------------------------#
            #   获得学习率下降的公式
            # ---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(
                lr_decay_type=lr_decay_type,
                lr=Init_lr_fit,
                min_lr=Min_lr_fit,
                total_iters=UnFreeze_Epoch,
            )

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            if distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(
                dataset=train_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=train_sampler,
            )
            gen_val = DataLoader(
                dataset=val_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=val_sampler,
            )
            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            UnFreeze_Epoch,
            Cuda,
            dice_loss,
            focal_loss,
            cls_weights,
            num_classes,
            fp16,
            scaler,
            save_period,
            save_dir,
            local_rank,
        )

        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="pytorch deeplabv3plus(xception) training"
    )
    # 模型和训练的参数
    parser.add_argument("--data-path", default="../../dataset/SUIMdevkit", help="dataset root")
    parser.add_argument(
        "--model-path",
        default="",
        help="model weights path",
    )
    parser.add_argument("--num-classes", default=7, type=int)
    # parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument(
        "--amp",
        default=True,
        type=bool,
        help="Use torch.cuda.amp for mixed precision training",
    )

    # 模型训练超参数 冻结参数训练
    parser.add_argument("--freeze-train", default=False, type=bool)
    parser.add_argument("-bf", "--batch-size-freeze", default=4, type=int)
    parser.add_argument("-buf", "--batch-size-unfreeze", default=4, type=int)
    parser.add_argument(
        "--epochs-freeze",
        default=0,
        type=int,
        metavar="N",
        help="number of freeze epochs to train",
    )
    parser.add_argument(
        "--epochs-unfreeze",
        default=500,
        type=int,
        metavar="N",
        help="number of unfreeze epochs to train",
    )
    parser.add_argument("--resume", default="False", help="resume from checkpoint")
    parser.add_argument(
        "--init-epoch", default=0, type=int, metavar="N", help="init epoch"
    )

    # 模型优化器超参数 Adam, SGD优化器
    parser.add_argument(
        "--init-lr", default=1e-2, type=float, help="initial learning rate"
    )  # adam: 5e-4, sgd: 7e-3
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--save-freq", default=5, type=int, help="save frequency")

    # 损失函数类型
    parser.add_argument("--dice-loss", default=False, type=bool)
    parser.add_argument("--focal-loss", default=False, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
