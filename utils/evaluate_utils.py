
# Import modules
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.metrics import metric


def evaluate_results(
        model_class, architecture, test_type="test", n_figures=10):
    """Compute metric results and display 'n_figures' prediction plots.

    Args:
        model_class (nn.Module): pytorch neural net.
        architecture (str): model architecture name
        test_type (str, optional): test type (train, val or test).
                                   Defaults to "test".
        n_figures (int, optional): num. of inference to be displayed.
                                   Defaults to 10.
    """
    # Test dataset
    _, loader = model_class.get_dataset(flag=test_type)

    # Evaluation on the test set
    inputs = []
    trues = []
    preds = []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(loader)):
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        pred, true = model_class.process_one_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark)

        inputs.append(batch_x[0, :, :].detach().numpy())
        trues.append(true[0, :, :].detach().numpy())
        preds.append(pred[0, :, :].detach().numpy())

    # Metrics
    mae, mse, rmse, mape, mspe = metric(np.array(preds), np.array(trues))
    print(f"{test_type} metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"MSPE: {mspe:.4f}")

    # Results visualization
    idxs = random.sample(range(0, len(preds)), n_figures)
    output_path = f"./output/{architecture}/{test_type}"
    os.makedirs(output_path, exist_ok=True)

    for i in range(n_figures):
        idx = idxs[i]
        plt.figure(figsize=(25, 5))
        plt.plot(np.concatenate((inputs[idx], preds[idx])), label="pred")
        plt.plot(np.concatenate((inputs[idx], trues[idx])), label="true")
        plt.legend()
        plt.savefig(os.path.join(output_path, f"pred_{i}.png"))
        plt.close()
