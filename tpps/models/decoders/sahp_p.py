import torch as th
import torch.nn as nn

from tpps.pytorch.activations import ParametricSoftplus

from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.base.monte_carlo import MCDecoder
from tpps.models.base.process import Events

from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon

class SAHP_P(MCDecoder):
    """
    The SAHP+ model. 
    """
    def __init__(
            self,
            n_mixture: int, 
            # MLP
            units_mlp: List[int],
            # Other params
            mc_prop_est: Optional[float] = 1.,
            emb_dim: Optional[int] = 2,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            mark_activation: Optional[str] = 'relu',
            hist_time_grouping: Optional[str] = 'summation',
            name:Optional[str] = 'thp+', 
            **kwargs):
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        super(SAHP_P, self).__init__(
            name=name,
            input_size=units_mlp[0],
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mu = nn.Linear(in_features=units_mlp[0], out_features=n_mixture, bias=False)
        self.eta = nn.Linear(in_features=units_mlp[0], out_features=n_mixture, bias=False)
        self.gamma = nn.Linear(in_features=units_mlp[0], out_features=n_mixture, bias=False)
        self.activation = nn.GELU()
        self.activation_gamma = ParametricSoftplus(units=n_mixture)
        self.final_activation = ParametricSoftplus(units=n_mixture)
        self.marks2 = nn.Linear(
            in_features=units_mlp[1], out_features=marks)
        self.hist_time_grouping = hist_time_grouping
        if self.hist_time_grouping == 'summation':
            self.marks1 = nn.Linear(
            in_features=units_mlp[0], out_features=units_mlp[1])
            self.mark_time = nn.Linear(
                in_features=self.encoding_size, out_features=units_mlp[1]
            )
        elif self.hist_time_grouping == 'concatenation':
            self.mark_time = nn.Linear(
                in_features=self.encoding_size + self.input_size, out_features=units_mlp[1]
            )
        self.mark_activation = self.get_mark_activation(mark_activation)



    def log_ground_intensity(
            self,
            query: th.Tensor,
            prev_times: th.Tensor,
            history_representations: th.Tensor,
            intensity_mask:Optional[th.Tensor] = None):
        
        mu = self.activation(self.mu(history_representations)) #[B,T,M]
        eta = self.activation(self.eta(history_representations))
        gamma = self.activation_gamma(self.gamma(history_representations))
        
        delta_t = (query - prev_times) * intensity_mask
        delta_t = delta_t.float()
        delta_t = delta_t + epsilon(dtype=delta_t.dtype, device=delta_t.device)
        delta_t = delta_t.unsqueeze(-1)

        outputs = mu + (eta -mu)*th.exp(-gamma*(delta_t)) #[B,T,M]

        outputs = self.final_activation(outputs)

        outputs = outputs + epsilon(dtype=outputs.dtype, device=outputs.device)

        outputs = th.sum(outputs, dim=-1)
    
        return th.log(outputs)


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
        
        (query_representations,
         intensity_mask) = self.get_query_representations(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations, 
            representations_mask=representations_mask)  # [B,T,enc_size], [B,T]

        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]

        log_ground_intensity = self.log_ground_intensity(
                                        query=query, 
                                        prev_times=prev_times, 
                                        history_representations=history_representations,
                                        intensity_mask=intensity_mask)
        
        log_mark_pmf = self.log_mark_pmf(
                        query_representations=query_representations, 
                        history_representations=history_representations)

        ground_intensity_integral = self.intensity_integral(
                                                query=query, 
                                                prev_times=prev_times,
                                                prev_times_idxs=prev_times_idxs,
                                                intensity_mask=intensity_mask,
                                                representations=representations
                                                )

        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        artifacts = {}
        artifacts['last_h'] = last_h.detach().cpu().numpy()
        

        return log_ground_intensity, log_mark_pmf, ground_intensity_integral, intensity_mask, artifacts
    
    def log_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        pass 

    def get_mark_activation(self, mark_activation):
        if mark_activation == 'relu':
            mark_activation = th.relu
        elif mark_activation == 'tanh':
            mark_activation = th.tanh
        elif mark_activation == 'sigmoid':
            mark_activation = th.sigmoid
        return mark_activation