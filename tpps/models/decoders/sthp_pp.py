import torch as th


from typing import List, Optional, Dict

from tpps.models.decoders.sthp import STHP
from tpps.models.base.process import Events


from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor


class STHP_PP(STHP):
    """
    The STHP++ model. 
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            # Other params
            mc_prop_est: Optional[float] = 1.,
            emb_dim: Optional[int] = 2,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            name: Optional[str] = 'shtp++',
            **kwargs):
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        super(STHP_PP, self).__init__(
            name=name,
            units_mlp=units_mlp,
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        
    
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

        b,l = query.shape
        representations_time = representations[0:b,:,:]
        representations_mark = representations[b:,:,:]

        history_representations_time = take_3_by_2(                            
            representations_time, index=prev_times_idxs)                   # [B,T,D]
        history_representations_mark = take_3_by_2(                          
            representations_mark, index=prev_times_idxs)

        
        prev_times = prev_times + epsilon(dtype=prev_times.dtype, device=prev_times.device)
        delta_t = query - prev_times
        check_tensor(delta_t)
        delta_t = delta_t.unsqueeze(-1)
        
        log_ground_intensity = self.log_ground_intensity(
                                        query=query, 
                                        prev_times=prev_times, 
                                        history_representations=history_representations_time)
        
        log_mark_pmf = self.log_mark_pmf(
                        delta_t=delta_t, 
                        history_representations=history_representations_mark)

        ground_intensity_integral = self.intensity_integral(
                                                query=query, 
                                                prev_times=prev_times,
                                                prev_times_idxs=prev_times_idxs,
                                                intensity_mask=intensity_mask,
                                                representations=representations_time
                                                )

        
        check_tensor(log_ground_intensity)
        check_tensor(log_mark_pmf)
        check_tensor(ground_intensity_integral * intensity_mask, positive=True)
        
        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h_t = history_representations_time[th.arange(batch_size), last_event_idx,:]
        last_h_m = history_representations_mark[th.arange(batch_size), last_event_idx,:] #[B,D]
        artifacts = {}
        artifacts['last_h_t'] = last_h_t.detach().cpu().numpy()
        artifacts['last_h_m'] = last_h_m.detach().cpu().numpy()


        return log_ground_intensity, log_mark_pmf, ground_intensity_integral, intensity_mask, artifacts