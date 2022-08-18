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

    report_name = "test_{}_{}_{}_{}_{}_{}_{}.csv".format(
        backbone_name, year, month, day, hour, minute, second)

    print("Saving Report('{}')".format(os.path.join(saving_path, report_name)))

    df.to_csv(os.path.join(saving_path, report_name))


def _main(args):
    # Hardware
    device = torch.device(
        "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # Data Path
    num_categories = 196

    # - Train
    test_base_dir, test_df_path = args.test_base_dir, args.test_df_path

    mean, std = [0.471, 0.460, 0.455], [0.267, 0.266, 0.271]

    # - Validation

    # val_transforms = transforms.Compose([transforms.Resize((args.input_width, args.input_height)),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize((args.input_width, args.input_height)),
                                         transforms.ToTensor(),
                                          #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         transforms.Normalize(mean, std)
                                          ])

    test_dataloader = data.StanfordCar196(
        img_base_dir=test_base_dir,
        annot_df_path=test_df_path,
        transformation=test_transforms,
        num_categories=num_categories
    )

    # CNN Backbone
    backbone = models.efficientnet_b0(
        pretrained=args.pretrained, progress=True)

    net = nets.MyEfficientNetB(
        eff_net_backbone=backbone,
        num_categories=num_categories
    )

    # Load the Model
    # net, _ = nets.load(
    #     ckpt_path=args.ckpt_load, model=net, optimizer=None)

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
    report, test_acc = dl.eval(
        net=net,
        dataloader=test_dataloader,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu=args.gpu,
    )

    print("Test Accuracy: {:.2f}".format(test_acc))

    save_report(df=report, backbone_name="efficientnet_b0",
                saving_path=args.report)


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
