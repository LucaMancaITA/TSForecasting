
# Import modules
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from utils.classes.data_args import DataArgs
from utils.classes.train_eval_args import TrainEvalArgs
from utils.classes.model_args import LstmArgs, InformerArgs
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

# Test dataset
_, loader = model.get_dataset(flag="test")

# Evaluation on the test set
inputs = []
trues = []
preds = []
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(loader)):
    batch_x = batch_x.float()
    batch_y = batch_y.float()

    pred, true = model.process_one_batch(
        batch_x, batch_y, batch_x_mark, batch_y_mark)

    inputs.append(batch_x[0, :, :].detach().numpy())
    trues.append(true[0, :, :].detach().numpy())
    preds.append(pred[0, :, :].detach().numpy())

pred_arr = np.array(preds)
pred_arr = np.reshape(pred_arr, (-1, 1))
true_arr = np.array(trues)
true_arr = np.reshape(true_arr, (-1, 1))

# Save the test inference
output_path = f"./output/{eval_args.architecture}/"
os.makedirs(output_path, exist_ok=True)

plt.figure(figsize=(25,5))
plt.plot(pred_arr, label="pred")
plt.plot(true_arr, label="true")
plt.legend()
plt.savefig(os.path.join(output_path, "test_inference.png"))
plt.close()
