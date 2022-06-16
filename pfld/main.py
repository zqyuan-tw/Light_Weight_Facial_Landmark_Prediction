import json
import torch
from trainer import Trainer, NME
from face import *
from argparse import ArgumentParser
from torchvision import transforms
from model import *
torch.manual_seed(777)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--do_train",
        help="Path to training configuration.",
        default="./config/pfld.json"
    )
    parser.add_argument(
        "--do_predict",
        help="Path to testing configuration..",
        # default="./config/predict.json"
    )
    parser.add_argument(
        "--pause",
        help="Path to start from pause configuration..",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.do_train is not None:
        with open(args.do_train) as f:
            config = json.load(f)
        # train_transform = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), transforms.ToTensor()])
        train_transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.GaussianBlur(3),
            transforms.ToTensor(), 
            RandomMask(0.1)
        ])
        # train_dataset = FaceDataset(config["training"]["data"], train_transform, coordinate_transform=transforms.Compose([RandomFlip(), RandomRotate(15)]))
        train_dataset = FaceDataset(config["training"]["data"], train_transform)

        # model = get_model(config["model"])
        # coefficient = 0.25
        coefficient = 1
        num_of_channels = num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient), int(128 * coefficient)]
        model = MobileNetV2(num_of_channels=num_of_channels)
        train_dataset = FaceDataset(config["training"]["data"], train_transform)
        batch_size = config["training"]["batch_size"]
        num_epoch = config["training"]["num_epoch"]
        lr = config["training"]["learning_rate"]
        saved_epoch = config["training"]["checkpoint"]
        saved_dir = config["saved_directory"]

        if config["validation"] is not None:
            val_dataset = FaceDataset(config["validation"]["data"], transforms.Compose([transforms.Resize((192, 192)), transforms.ToTensor()]))
            val_epoch = config["validation"]["epoch"]
        else:
            val_dataset = None
            val_epoch = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 40, 45], gamma=0.1)
        criterion = NME()

        trainer = Trainer(model, device)
        trainer.fit(train_dataset, batch_size, optimizer, criterion, num_epoch, saved_epoch, scheduler, saved_dir, val_epoch, val_dataset)

    if args.do_predict is not None:
        with open(args.do_predict) as f:
            config = json.load(f)

        # coefficient = 0.25
        coefficient = 1
        num_of_channels = num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient), int(128 * coefficient)]
        model = MobileNetV2(num_of_channels=num_of_channels)
        test_dataset = FaceDataset(
            config["testing"]["data"], transforms.Compose([transforms.Resize((192, 192)), transforms.ToTensor()]), do_train=False)
        batch_size = config["testing"]["batch_size"]
        saved_path = config["saved_path"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = Trainer(model, device)
        trainer.load_from_pretrained(saved_path)
        ids, lms = trainer.predict(test_dataset, batch_size)
        with open('solution.txt', 'w') as f:
            for i in range(len(ids)):
                f.write(f"{ids[i]} {' '.join(str(x) for x in lms[i])}\n")

    if args.pause is not None:
        with open(args.do_train) as f:
            config = json.load(f)
        # train_transform = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), transforms.ToTensor()])
        train_transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            # transforms.GaussianBlur(3),
            transforms.ToTensor(), 
            # RandomMask(0.1)
        ])
        # train_dataset = FaceDataset(config["training"]["data"], train_transform, coordinate_transform=transforms.Compose([RandomFlip(), RandomRotate(15)]))
        train_dataset = FaceDataset(config["training"]["data"], train_transform)

        # model = get_model(config["model"])
        coefficient = 0.25
        num_of_channels = num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient), int(128 * coefficient)]
        model = MobileNetV2(num_of_channels=num_of_channels)
        train_dataset = FaceDataset(config["training"]["data"], train_transform)
        batch_size = config["training"]["batch_size"]
        num_epoch = config["training"]["num_epoch"]
        lr = config["training"]["learning_rate"]
        saved_epoch = config["training"]["checkpoint"]
        saved_dir = config["saved_directory"]
        saved_model = config

        if config["validation"] is not None:
            val_dataset = FaceDataset(config["validation"]["data"], transforms.ToTensor())
            val_epoch = config["validation"]["epoch"]
        else:
            val_dataset = None
            val_epoch = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        criterion = NME()

        trainer = Trainer(model, device)
        trainer.load_from_pretrained(saved_model)
        trainer.fit(train_dataset, batch_size, optimizer, criterion, num_epoch, saved_epoch, scheduler, saved_dir, val_epoch, val_dataset)

