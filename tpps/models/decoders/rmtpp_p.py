import torch as th
import torch.nn as nn

from typing import Dict, Optional, List

from tpps.models.decoders.rmtpp import RMTPPDecoder
from tpps.utils.events import Events

from tpps.utils.stability import epsilon

class RMTPP_P(RMTPPDecoder):
    """
    The RMTPP+ model. 
    """
    def __init__(
            self,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            mark_activation: Optional[str] = 'relu',
            name: Optional[str] = 'rmtpp+',
            **kwargs):
        super(RMTPP_P, self).__init__(
            name=name,
            units_mlp=units_mlp,
            multi_labels=multi_labels,
            marks=marks,
            encoding=encoding,
            mark_activation=mark_activation,
            **kwargs)
        self.mark_time = nn.Linear(
                in_features=self.encoding_size + self.input_size, out_features=units_mlp[1])

        
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