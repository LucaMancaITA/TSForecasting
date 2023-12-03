
# Import modules
import os
import json

import torch

from utils.classes.data_args import DataArgs
from utils.classes.train_eval_args import TrainEvalArgs
from utils.classes.model_args import LstmArgs, InformerArgs
from utils.evaluate_utils import evaluate_results
from modelling.lstm_model import LstmModel
from modelling.informer_model import InformerModel


# Read config files
config_path = "./config/evaluate/args.json"
with open(config_path, "r", encoding="utf-8") as file:
    args = json.load(file)

# Data and Eval args
data_args = DataArgs(**args["data"])
eval_args = TrainEvalArgs(**args["evaluate"])

# Initialize model class
if eval_args.architecture == "lstm":
    net_path = "./config/models/lstm.json"
    with open(net_path, "r", encoding="utf-8") as file:
        net_args = json.load(file)
    model_args = LstmArgs(**net_args)

    model = LstmModel(data_args, eval_args, model_args)

elif eval_args.architecture == "informer":
    net_path = "./config/models/informer.json"
    with open(net_path, "r", encoding="utf-8") as file:
        net_args = json.load(file)
    model_args = InformerArgs(**net_args)

    model = InformerModel(data_args, eval_args, model_args)

# Load checkpoint
model_path = os.path.join(
    eval_args.checkpoints, eval_args.model_name, "checkpoint.pth")
model.model.load_state_dict(torch.load(model_path))

# Train
evaluate_results(
    model, eval_args.architecture,
    test_type="train", n_figures=args["outputs"]["n_figures"])

# Val
evaluate_results(
    model, eval_args.architecture,
    test_type="val", n_figures=args["outputs"]["n_figures"])

# Test
evaluate_results(
    model, eval_args.architecture,
    test_type="test", n_figures=args["outputs"]["n_figures"])
