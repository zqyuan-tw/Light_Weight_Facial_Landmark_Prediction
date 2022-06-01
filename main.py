import json
import torch
from trainer import Trainer, NME
from face import FaceDataset
from argparse import ArgumentParser
from torchvision import transforms
from models import ConvNet

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--do_train",
        help="Path to training configuration.",
    )
    parser.add_argument(
        "--do_predict",
        help="Path to testing configuration..",
    )   

    args = parser.parse_args()

    return args

def get_model(model):
    if model == "ConvNet":
        return ConvNet()
    if model == "":
        return 

if __name__ == "__main__":
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.do_train is not None:
        with open(args.do_train) as f:
            config = json.load(f)

        model = get_model(config["model"])
        train_dataset = FaceDataset(config["training"]["data"], transforms.ToTensor())
        batch_size = config["training"]["batch_size"]
        num_epoch = config["training"]["num_epoch"]
        lr = config["training"]["learning_rate"]
        saved_epoch = config["training"]["checkpoint"]
        saved_dir = config["saved_directory"]

        if config["validation"] is not None:
            val_dataset = FaceDataset(config["validation"]["data"], transforms.ToTensor())
            val_epoch = config["validation"]["epoch"]
        else:
            val_dataset = None
            val_epoch = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        criterion = NME()

        trainer = Trainer(model, device)
        trainer.fit(train_dataset, batch_size, optimizer, criterion, num_epoch, saved_epoch, scheduler, saved_dir, val_epoch, val_dataset)

    if args.do_predict is not None:
        # TODO
        pass
        