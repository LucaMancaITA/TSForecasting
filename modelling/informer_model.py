
# Import modules
import torch

from modelling.base_model import BaseModel
from net.informer.informer import Informer


class InformerModel(BaseModel):
    """Informer model."""
    def __init__(self, data_args, train_eval_args, informer_args):
        super().__init__(data_args, train_eval_args)
        self.model_args = informer_args

        self.model = self.build_model()

    def build_model(self):
        model = Informer(
            enc_in=self.model_args.enc_in,
            dec_in=self.model_args.dec_in,
            c_out=self.model_args.c_out,
            seq_len=self.data_args.input_len,
            label_len=self.data_args.label_len,
            out_len=self.data_args.pred_len,
            d_model=self.model_args.d_model,
            n_heads=self.model_args.n_heads,
            e_layers=self.model_args.e_layers,
            d_layers=self.model_args.d_layers,
            d_ff=self.model_args.d_ff
        )

        num_params = self.count_parameters(model)
        print(f"Number of parameters: {num_params}")

        return model

    def process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # decoder input
        dec_inp = torch.zeros(
            [batch_y.shape[0], self.data_args.pred_len, batch_y.shape[-1]]
        ).float()
        dec_inp = torch.cat(
            [batch_y[:,:self.data_args.label_len,:], dec_inp], dim=1
        ).float()
        # encoder - decoder
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = 0
        batch_y = batch_y[:,-self.data_args.pred_len:,f_dim:]

        return outputs, batch_y
