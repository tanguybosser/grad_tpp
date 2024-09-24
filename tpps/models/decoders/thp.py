import torch as th
import torch.nn as nn

from tpps.pytorch.activations import ParametricSoftplus

from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.base.monte_carlo import MCDecoder
from tpps.models.base.process import Events


from tpps.utils.encoding import encoding_size
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor


class THP(MCDecoder):
    """
    The base THP model. See https://arxiv.org/abs/2002.09291.pdf
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
            name: Optional[str] = 'thp',
            **kwargs):
        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        super(THP, self).__init__(
            name=name,
            input_size=units_mlp[0],
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.w_t = nn.Linear(in_features=1, out_features=marks)
        self.w_h = nn.Linear(in_features=units_mlp[0], out_features=marks)
        self.activation = ParametricSoftplus(units=marks)

    def log_intensity(
            self,
            query: th.Tensor,
            prev_times: th.Tensor,
            history_representations: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the log_intensity and a mask


        """
        prev_times = prev_times + epsilon(dtype=prev_times.dtype, device=prev_times.device)
        delta_t = query - prev_times
        check_tensor(delta_t)
        delta_t = delta_t.unsqueeze(-1).float()
        w_delta_t = self.w_t(delta_t)
        w_history = self.w_h(history_representations)
        outputs = self.activation(w_delta_t + w_history)
        outputs = outputs + epsilon(dtype=outputs.dtype, device=outputs.device)
        return th.log(outputs)

    def log_ground_intensity(self,
            query: th.Tensor,
            prev_times: th.Tensor,
            history_representations: th.Tensor,
            intensity_mask:Optional[th.Tensor] = None):

        log_marked_intensity = self.log_intensity(
                                        query=query,
                                        prev_times=prev_times,
                                        history_representations=history_representations)

        ground_intensity = th.sum(th.exp(log_marked_intensity), dim=-1)

        return th.log(ground_intensity)



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

        log_marked_intensity = self.log_intensity(
                                                query=query,
                                                prev_times=prev_times,
                                                history_representations=history_representations)
        
        marked_intensity = th.exp(log_marked_intensity)

        ground_intensity = th.sum(marked_intensity, dim=-1)
        log_ground_intensity = th.log(ground_intensity)
        
        mark_pmf = marked_intensity / ground_intensity.unsqueeze(-1)
        log_mark_pmf = th.log(mark_pmf)

        ground_intensity_integral = self.intensity_integral(
                                            query=query, 
                                            prev_times=prev_times,
                                            prev_times_idxs=prev_times_idxs,
                                            intensity_mask=intensity_mask,
                                            representations=representations)
                                            
        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        artifacts = {}
        artifacts['last_h'] = last_h.detach().cpu().numpy()

        return log_ground_intensity, log_mark_pmf, ground_intensity_integral, intensity_mask, artifacts