import torch as th
import torch.nn as nn


from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.fnn import FNN
from tpps.models.decoders.base.decoder import Decoder
from tpps.models.base.process import Events


from tpps.utils.index import take_3_by_2
from tpps.utils.stability import check_tensor

from tpps.utils.nnplus import non_neg_param

class FNN_D(Decoder):
    """
    The FNN-D model. 
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = "nonneg",
            activation_final_mlp: Optional[str] = "parametric_softplus",
            # Other params
            model_log_cm: Optional[bool] = False,
            do_zero_subtraction: Optional[bool] = True,
            emb_dim: Optional[int] = 2,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            name: Optional[str] = 'fnn-d',
            **kwargs):
        super(FNN_D, self).__init__(
            name=name,
            input_size=units_mlp[0],
            marks=marks,
            **kwargs)
        self.model_time = FNN(
            units_mlp=units_mlp,
            activation_mlp=activation_mlp,
            dropout_mlp=dropout_mlp,
            constraint_mlp=constraint_mlp,
            activation_final_mlp=activation_final_mlp,
            model_log_cm=model_log_cm,
            do_zero_subtraction=do_zero_subtraction,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs
        )
        self.model_mark = FNN(
            units_mlp=units_mlp,
            activation_mlp=activation_mlp,
            dropout_mlp=dropout_mlp,
            constraint_mlp=constraint_mlp,
            activation_final_mlp=activation_final_mlp,
            model_log_cm=model_log_cm,
            do_zero_subtraction=do_zero_subtraction,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs
        )
        self.input_size = self.model_time.input_size

    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[bool] = None, 
            sampling: Optional[bool] = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        
        
        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D] 
        
        self.model_time.mu.data = non_neg_param(self.model_time.mu.data)
        self.model_mark.mu.data = non_neg_param(self.model_mark.mu.data)

        check_tensor(self.model_time.mu.data, positive=True)
        check_tensor(self.model_mark.mu.data, positive=True)

        query.requires_grad = True
        intensity_integrals_time, intensity_mask = self.model_time.diff_cum_marked_intensity(
                            events=events,
                            query=query,
                            prev_times=prev_times,
                            prev_times_idxs=prev_times_idxs,
                            pos_delta_mask=pos_delta_mask,
                            is_event=is_event,
                            representations=representations,
                            representations_mask=representations_mask
                    )
        
        marked_intensity_time = self.model_time.marked_intensity(
                                    query=query, 
                                    intensity_integrals=intensity_integrals_time
        )
        check_tensor(marked_intensity_time, positive=True, strict=True)
        
        intensity_integrals_mark, intensity_mask = self.model_mark.diff_cum_marked_intensity(
                            events=events,
                            query=query,
                            prev_times=prev_times,
                            prev_times_idxs=prev_times_idxs,
                            pos_delta_mask=pos_delta_mask,
                            is_event=is_event,
                            representations=representations,
                            representations_mask=representations_mask
                    )

        marked_intensity_mark = self.model_time.marked_intensity(
                                    query=query, 
                                    intensity_integrals=intensity_integrals_mark
        )

        query.requires_grad = False

        check_tensor(marked_intensity_mark, positive=True, strict=True)
        
        ground_intensity = th.sum(marked_intensity_time, dim=-1)
        log_ground_intensity = th.log(ground_intensity)

        mark_pmf = marked_intensity_mark / th.sum(marked_intensity_mark, dim=-1).unsqueeze(-1)
        log_mark_pmf = th.log(mark_pmf)
        
        ground_intensity_integrals = th.sum(intensity_integrals_time, dim=-1)

        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        artifacts = {}
        artifacts['last_h'] = last_h.detach().cpu().numpy()
            
        return (log_ground_intensity,
                log_mark_pmf,
                ground_intensity_integrals, 
                intensity_mask,
                artifacts)  # [B,T,M], [B,T,M], [B,T], Dict
