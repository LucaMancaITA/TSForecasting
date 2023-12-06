# TSForecasting

Time Series Forecasting model benchmarking for educational purpose.

The idea is collect here the state-of-the-art models used for time series prediction.
Currently, the repository includes the following architectures:
* LSTM [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
* Informer [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
* Autoformer [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)


## Usage

The dataset must be loaded inside a `data/` directory.

To train a new model run the `train.py` script. All the training variables can be set inside the `config/training/args.json`, whereas the model hyperparameters can be set in the `config/models/` related config file.
Once a model is trained, the tensorboard logs will be stored inside the `runs/` directory, while the model will be saved in the `checkpoints/` directory.

Once the model is trained, it possible to perform model evaluation by running `evaluate.py`, which computes the metrics (MAE, MSE, RMSE, MAPE, MSPE) on the training, validation and test set, as well as generating some inference plots (stored in the `output/` folder).
Finally, by running `inference.py`, the model perform the inference on the entire test set, using a sliding window equal to the prediction length, and store the time-series predicted and ground truth in the `output/` directory.

A quick training iteration has been done on the [ETTh](https://paperswithcode.com/dataset/ett) dataset; results on the test set are collected in the following table (`input_len` is set to 96 and `output_len` to 24, equal to 1 day):

| Architecture  | Num. of parameters | MAE   | MSE   | RMSE  |
| ------------- | ------------------ | ----- | ----- | ----- |
| LSTM          | $\approx$ 220000   | 0.475 | 0.336 | 0.580 |
| Informer      | $\approx$ 260000   | 0.349 | 0.188 | 0.433 |
| Autoformer    | $\approx$ 260000   | 0.268 | 0.128 | 0.358 |

*Note: models architecture are intentionally kept small to allow a faster training, even without a GPU.*
