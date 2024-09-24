import torch as th
import torch.nn as nn


from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.sahp import SAHP
from tpps.models.decoders.base.decoder import Decoder
from tpps.models.base.process import Events


from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import check_tensor


class SAHP_D(Decoder):
    """
    The SAHP-D model. 
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = None,
            activation_final_mlp: Optional[str] = "parametric_softplus",
            # Other params
            mc_prop_est: Optional[float] = 1.,
            emb_dim: Optional[int] = 2,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1, 
            name: Optional[str] = 'sahp-d',
            **kwargs):
        super(SAHP_D, self).__init__(
            name=name,
            input_size=units_mlp[0],
            marks=marks,
            **kwargs)
        self.model_time = SAHP(
            units_mlp=units_mlp,
            activation_mlp=activation_mlp,
            dropout_mlp=dropout_mlp,
            constraint_mlp=constraint_mlp,
            activation_final_mlp=activation_final_mlp,
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks, 
            **kwargs
        )
        self.model_mark = SAHP(
            units_mlp=units_mlp,
            activation_mlp=activation_mlp,
            dropout_mlp=dropout_mlp,
            constraint_mlp=constraint_mlp,
            activation_final_mlp=activation_final_mlp,
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks, 
            **kwargs
        )

    def forward(self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None):
        
        
        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]

        log_ground_intensity = self.model_time.log_ground_intensity(
                                                    query=query,
                                                    prev_times=prev_times,
                                                    history_representations=history_representations,
                                                    intensity_mask=intensity_mask
                                                )
        
        ground_intensity_integral = self.model_time.intensity_integral(
                                                query=query, 
                                                prev_times=prev_times,
                                                prev_times_idxs=prev_times_idxs,
                                                intensity_mask=intensity_mask,
                                                representations=representations
                                            )

        log_marked_intensity = self.model_mark.log_intensity(
                                            query=query,
                                            prev_times=prev_times,
                                            history_representations=history_representations,
                                            intensity_mask=intensity_mask
                                        )
        
        marked_intensity = th.exp(log_marked_intensity)
        log_mark_pmf = self.model_mark.log_mark_pmf(marked_intensity)

        check_tensor(log_ground_intensity * intensity_mask)
        check_tensor(ground_intensity_integral * intensity_mask, positive=True)
        check_tensor(log_mark_pmf * intensity_mask.unsqueeze(-1))

        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        artifacts = {}
        artifacts['last_h'] = last_h.detach().cpu().numpy()
        

        return log_ground_intensity, log_mark_pmf, ground_intensity_integral, intensity_mask, artifacts
