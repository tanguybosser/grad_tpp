import abc

import torch as th
import torch.nn as nn

from typing import Optional

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder

from tpps.utils.stability import epsilon

class CumulativeDecoder(VariableHistoryDecoder, abc.ABC):
    """Decoder based on Cumulative intensity method. Here, the cumulative
       intensity is specified, but its derivative is directly computed

    Args:
        name: The name of the decoder class.
        do_zero_subtraction: If `True` the class computes
            Lambda(tau) = Lambda'(tau) - Lambda'(0)
            in order to enforce Lambda(0) = 0. Defaults to `True`.
        input_size: The dimensionality of the input required from the encoder.
            Defaults to `None`. This is mainly just for tracking/debugging
            ease.
        emb_dim: Size of the embeddings. Defaults to 1.
        encoding: Way to encode the queries: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self,
                 name: str,
                 do_zero_subtraction: Optional[bool] = True,
                 model_log_cm: Optional[bool] = False,
                 input_size: Optional[int] = None,
                 emb_dim: Optional[int] = 1,
                 encoding: Optional[str] = "times_only",
                 time_encoding: Optional[str] = "relative",
                 marks: Optional[int] = 1,
                 **kwargs):
        super(CumulativeDecoder, self).__init__(
            name=name,
            input_size=input_size,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.do_zero_subtraction = do_zero_subtraction
        self.model_log_cm = model_log_cm
    
    def reset_parameters(self):
        nn.init.uniform_(self.mu)
    
    def log_mark_pmf(
            self, 
            query_representations:th.Tensor, 
            history_representations:th.Tensor):
        
        history_times = th.cat((history_representations, query_representations), dim=-1)
        p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.mark_time(history_times))), dim=-1)
        
        p_m = p_m + epsilon(dtype=p_m.dtype, device=p_m.device)

        return th.log(p_m)