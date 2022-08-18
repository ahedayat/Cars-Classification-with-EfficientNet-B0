import os
import utils as utils
import losses as losses
from datetime import datetime, date


import torch
import nets as nets
import deeplearning as dl
import dataloaders as data
import torch.optim as optim
from torchvision import transforms, models


def save_report(df, backbone_name, saving_path):
    """
        Saving Output Report Dataframe that is returned in Training
    """
    _time = datetime.now()
    hour, minute, second = _time.hour, _time.minute, _time.second

    _date = date.today()
    year, month, day = _date.year, _date.month, _date.day

    report_name = "{}_{}_{}_{}_{}_{}_{}.csv".format(
        backbone_name, year, month, day, hour, minute, second)

    df.to_csv(os.path.join(saving_path, report_name))


def _main(args):
    # Hardware
    device = torch.device(
        "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # Data Path
    num_categories = 196

    # - Train
    train_base_dir, train_df_path = args.train_base_dir, args.train_df_path
    # train_transforms = transforms.Compose([transforms.Resize((args.input_width, args.input_height)),
    #                                        transforms.RandomHorizontalFlip(),
    #                                        transforms.RandomRotation(15),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    mean, std = [0.471, 0.460, 0.455], [0.267, 0.266, 0.271]

    # train_transforms = transforms.Compose([
    #     transforms.Resize((args.input_width, args.input_height)),
    #     transforms.RandomCrop(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(
    #         brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std, inplace=True)
    # ])

    train_transforms = transforms.Compose([transforms.Resize((args.input_width, args.input_height)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(15),
                                           transforms.ToTensor(),
                                           #    transforms.Normalize([0.485, 0.456, 0.406], [
                                           #                         0.229, 0.224, 0.225])
                                           transforms.Normalize(mean, std)
                                           ])

    train_dataloader = data.StanfordCar196(
        img_base_dir=train_base_dir,
        annot_df_path=train_df_path,
        transformation=train_transforms,
        num_categories=num_categories,
        mode="train"
    )

    # - Validation

    # val_transforms = transforms.Compose([transforms.Resize((args.input_width, args.input_height)),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([transforms.Resize((args.input_width, args.input_height)),
                                         transforms.ToTensor(),
                                        #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         transforms.Normalize(mean, std)
                                         ])

    val_dataloader = data.StanfordCar196(
        img_base_dir=train_base_dir,
        annot_df_path=train_df_path,
        transformation=val_transforms,
        num_categories=num_categories,
        mode="val"
    )

    # CNN Backbone
    backbone = models.efficientnet_b0(
        pretrained=args.pretrained, progress=True)

    print("**** Freezing Backbone's Convoloutional Layers...")
    for param in backbone.parameters():
        param.requires_grad = False

    net = nets.MyEfficientNetB(
        eff_net_backbone=backbone,
        num_categories=num_categories
    )

    net.freeze_backbone()
    net.unfreeze_layers([6, 7, 8])

    # Optimizer
    assert args.optimizer in [
        "sgd", "adam"], "Optimizer must be one of this items: ['sgd', 'adam']"

    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Learning Rate Schedular
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='max',
                                                        patience=3,
                                                        threshold=0.9,
                                                        min_lr=1e-6,
                                                        verbose=True,
                                                        )

    # Loading Model
    if args.load_model is not None:
        print("**** Loading Model...")
        net, optimizer = nets.load(
            ckpt_path=args.load_model, model=net, optimizer=optimizer)

    if args.gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            net = net.cuda(device=device)

    # Loss Function
    assert args.criterion in [
        'cross_entropy', "label_smoothing_cross_entropy"], "Loss Function must be one of this items: [ 'cross_entropy', 'label_smoothing_cross_entropy' ]"
    if args.criterion == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    else:
        criterion = losses.LabelSmoothingCrossEntropy()

    # Checkpoint Address
    saving_path, saving_prefix = args.ckpt_path, args.ckpt_prefix
    saving_frequency = args.save_freq

    # Training
    net, report = dl.train(
        net=net,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        saving_path=saving_path,
        saving_prefix=saving_prefix,
        saving_frequency=saving_frequency,
        # saving_model_every_epoch=False,
        gpu=args.gpu,
        lr_scheduler=lr_scheduler
        # milestones=[80, 90, 110]
    )

    save_report(df=report, backbone_name="efficientnet_b0",
                saving_path=args.report)

    nets.save(
        file_path=saving_path,
        file_name="{}_final".format(saving_prefix),
        model=net,
        optimizer=optimizer
    )


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
