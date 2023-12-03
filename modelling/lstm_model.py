
# Import modules
from modelling.base_model import BaseModel
from net.lstm.lstm import Lstm


class LstmModel(BaseModel):
    """LSTM model."""
    def __init__(self, data_args, train_eval_args, model_args):
        super().__init__(data_args, train_eval_args)
        self.model_args = model_args

        self.model = self.build_model()

    def build_model(self):
        if self.data_args.features == "M":
            input_feat = 7
        else:
            input_feat = 1

        model = Lstm(
            input_feat=input_feat,
            pred_len=self.data_args.pred_len,
            hidden_units=self.model_args.hidden_units,
            num_layers=self.model_args.num_layers)

        num_params = self.count_parameters(model)
        print(f"Number of parameters: {num_params}")

        return model

    def process_one_batch(self, batch_x, batch_y,  batch_x_mark, batch_y_mark):
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)

        f_dim = 0
        batch_y = batch_y[:,-self.data_args.pred_len:,f_dim:]

        return outputs, batch_y
