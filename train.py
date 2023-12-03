
# Import modules
import json

from utils.classes.data_args import DataArgs
from utils.classes.train_eval_args import TrainEvalArgs
from utils.classes.model_args import LstmArgs, InformerArgs
from modelling.lstm_model import LstmModel
from modelling.informer_model import InformerModel


# Read config files
config_path = "./config/training/args.json"
with open(config_path, "r", encoding="utf-8") as file:
    args = json.load(file)

# Train and data arguments
data_args = DataArgs(**args["data"])
train_args = TrainEvalArgs(**args["train"])

# Build the model
if train_args.architecture == "lstm":
    net_path = "./config/models/lstm.json"
    with open(net_path, "r", encoding="utf-8") as file:
        net_args = json.load(file)
    model_args = LstmArgs(**net_args)

    model = LstmModel(data_args, train_args, model_args)

elif train_args.architecture == "informer":
    net_path = "./config/models/informer.json"
    with open(net_path, "r", encoding="utf-8") as file:
        net_args = json.load(file)
    model_args = InformerArgs(**net_args)

    model = InformerModel(data_args, train_args, model_args)

# Training loop
model.training()
