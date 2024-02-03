import os
import pickle
from dataset.Synapse import SynapseDataset, RandomGenerator
from dataset.ACDC import AcdcDataset
from dataset.KVASIR import kvasirDataset
# from dataset.SliceLoader import SliceDataset
import torch
from torchvision import transforms


def generate_dataset(args):
    if args.dataset != 'SYNAPSE' and args.dataset != 'synapse':
        split_dir = os.path.join(args.src_dir, "splits.pkl")
        with open(split_dir, "rb") as f:
            splits = pickle.load(f)
        tr_keys = splits[args.fold]['train']
        val_keys = splits[args.fold]['val']
        test_keys = splits[args.fold]['test']

        if args.tr_size < len(tr_keys):
            tr_keys = tr_keys[0:args.tr_size]

        print(tr_keys)
        print(val_keys)
        print(test_keys)


    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        train_ds = AcdcDataset(keys=tr_keys, mode='train', args=args)
        val_ds = AcdcDataset(keys=val_keys, mode='val', args=args)
        test_ds = AcdcDataset(keys=test_keys, mode='val', args=args)
    elif args.dataset == 'kvasir' or args.dataset == 'KVASIR':
        args.img_size = 320
        train_ds = kvasirDataset(keys=tr_keys, mode='train', args=args)
        val_ds = kvasirDataset(keys=val_keys, mode='val', args=args)
        test_ds = kvasirDataset(keys=test_keys, mode='val', args=args)
    elif args.dataset == 'synapse' or args.dataset == 'SYNAPSE':
        args.img_size = 224
        image_size = 1024
        vit_patch_size = 16
        img_embedding_size = image_size // vit_patch_size
        low_res = img_embedding_size * 4
        # train_ds = SynapseDataset(keys=tr_keys, split='train', args=args)
        # val_ds = SynapseDataset(keys=val_keys, split='val', args=args)
        # test_ds = SynapseDataset(keys=test_keys, split='val', args=args)
        train_ds = SynapseDataset(base_dir='../SAMed/train_npz_new_224/', list_dir='../SAMed/lists/lists_Synapse', split='train', transform=transforms.Compose(
                                   [RandomGenerator(output_size=[1024, 1024], low_res=[low_res, low_res])]), tr_size=args.tr_size)
        val_ds = None
        test_ds = None
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    if args.dataset != 'SYNAPSE' and args.dataset != 'synapse':
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
        )
    else: 
        val_loader = None

    if args.dataset != 'SYNAPSE' and args.dataset != 'synapse':
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=(test_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
        )
    else:
        test_loader = None

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler


def generate_test_loader(key, args):
    key = [key]
    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        test_ds = AcdcDataset(keys=key, mode='val', args=args)
    elif args.dataset == 'kvasir' or args.dataset == 'KVASIR':
        args.img_size = 320
        test_ds = kvasirDataset(keys=key, mode='val', args=args)
    elif args.dataset == 'synapse' or args.dataset == 'SYNAPSE':
        args.img_size = 512
        test_ds = SynapseDataset(keys=key, mode='val', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    # if args.distributed:
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    # else:
    #     test_sampler = None
    test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    )

    return test_loader


def generate_contrast_dataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]

    print(tr_keys)
    print(val_keys)

    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        train_ds = AcdcDataset(keys=tr_keys, mode='contrast', args=args)
        val_ds = AcdcDataset(keys=val_keys, mode='contrast', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    return train_loader, val_loader
