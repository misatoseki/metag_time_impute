import argparse
import torch
import datetime
import json
import yaml
import os,gc
import time
from sklearn.model_selection import KFold
import numpy as np

from main_model import CSDI_vatanen
from dataset_vatanen import get_dataloader,vatanen_Dataset
from utils import train, evaluate

time_start = time.time()

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--inputdir", type=str, default="../data/data_csdi")


args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config["train"]["batch_size"] = args.batch_size
config["train"]["epochs"] = args.epochs

print(json.dumps(config, indent=4))

with open(args.inputdir + "/attribute", "r") as f:
    attributes = f.read().split('\n')
    attributes.remove('')

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# only to obtain total length of dataset
dataset = vatanen_Dataset(eval_length=8, missing_ratio=args.testmissingratio, seed=args.seed, attributes=attributes, inputdir=args.inputdir)

mae_cv = []
kf = KFold(n_splits=5, shuffle=True, random_state=1)
for _fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    foldername = "./save/vatanen_" + current_time + "_MR-" + str(args.testmissingratio) + "_fold" + str(_fold) + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    train_loader, valid_loader, test_loader = get_dataloader(
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
        inputdir=args.inputdir, 
        attributes=attributes,
        foldername=foldername,
        train_index=train_index,
        test_index=test_index
    )
    
    model = CSDI_vatanen(config, args.device, len(attributes)).to(args.device)
    
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )

        output_path = foldername + "/model.pth"
        model.load_state_dict(torch.load(output_path)) 
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    
    mae = evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
    mae_cv.append(mae)
    
    print("\nelapsed time: {0}".format(time.time() - time_start) + " [sec]")   
 

print("Average MAE:", np.mean(mae_cv))    



