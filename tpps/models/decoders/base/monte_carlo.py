import abc

import torch as th

from typing import Optional, Tuple, Dict

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder
from tpps.models.base.process import Events
from tpps.utils.stability import check_tensor
from tpps.utils.index import take_3_by_2

class MCDecoder(VariableHistoryDecoder, abc.ABC):
    """Decoder based on Monte Carlo method. Here, the intensity is specified,
    but its cumulative function is determined by a Monte Carlo estimation.

    Args:
        name: The name of the decoder class.
        mc_prop_est: Proportion of numbers of samples for the MC method,
            compared to the size of the input. Defaults to 1.
        input_size: The dimensionality of the input required from the encoder.
            Defaults to `None`. This is mainly just for tracking/debugging
            ease.
        emb_dim: Size of the embeddings. Defaults to 1.
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the queries: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self,
                 name: str,
                 mc_prop_est: Optional[float] = 1.,
                 input_size: Optional[int] = None,
                 emb_dim: Optional[int] = 1,
                 temporal_scaling: Optional[float] = 1.,
                 encoding: Optional[str] = "times_only",
                 time_encoding: Optional[str] = "relative",
                 marks: Optional[int] = 1,
                 **kwargs):
        super(MCDecoder, self).__init__(
            name=name,
            input_size=input_size,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mc_prop_est = mc_prop_est

    @abc.abstractmethod
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


    @abc.abstractmethod
    def log_ground_intensity(
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
                
    def log_mark_pmf(
            self, 
            marked_intensity:th.Tensor
    ):
        mark_pmf = marked_intensity/th.sum(marked_intensity, dim=-1).unsqueeze(-1)
        log_mark_pmf = th.log(mark_pmf)
        check_tensor(log_mark_pmf)
        return log_mark_pmf
    
    def intensity_integral(
            self,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            intensity_mask:th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
            ):
        
        # Create Monte Carlo samples and sort them
        n_est = int(self.mc_prop_est)
        mc_times_samples = th.rand(
            query.shape[0], query.shape[1], n_est, device=query.device) * \
            (query - prev_times).unsqueeze(-1) + prev_times.unsqueeze(-1)
        mc_times_samples = th.sort(mc_times_samples, dim=-1).values
        mc_times_samples = mc_times_samples.reshape(
            mc_times_samples.shape[0], -1)  # [B, TxN]

        prev_times_idxs=th.repeat_interleave(
                prev_times_idxs, n_est, dim=-1)

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]
        
        mc_log_ground_intensity = self.log_ground_intensity(
            query=mc_times_samples,
            prev_times=th.repeat_interleave(prev_times, n_est, dim=-1),
            history_representations=history_representations,
            intensity_mask=th.repeat_interleave(intensity_mask, n_est, dim=-1))  # [B,TxN]

        mc_log_ground_intensity = mc_log_ground_intensity.reshape(
            query.shape[0], query.shape[1], n_est)  # [B,T,N]
        
        mc_log_ground_intensity = mc_log_ground_intensity * \
            intensity_mask.unsqueeze(-1)  # [B,T,N]
        ground_intensity_mc = th.exp(mc_log_ground_intensity)

        ground_intensity_integrals = (query - prev_times) * \
            ground_intensity_mc.sum(-1) / float(n_est)  # [B,T]
        
        return ground_intensity_integrals