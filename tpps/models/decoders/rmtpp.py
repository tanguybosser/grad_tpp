import torch as th
import torch.nn as nn
th.autograd.set_detect_anomaly(True)

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder
from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor
from tpps.utils.nnplus import non_neg_param



class RMTPPDecoder(VariableHistoryDecoder):
    """
    The base RMTPP model. See https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf.
    
    """
    def __init__(
            self,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            mark_activation: Optional[str] = 'relu',
            name: Optional[str] = 'rmtpp',
            **kwargs):
        super(RMTPPDecoder, self).__init__(
            name=name,
            input_size=units_mlp[0],
            encoding=encoding,
            marks=marks)
        self.w = nn.Parameter(th.Tensor(1))
        self.w_list = []
        self.w_t  = nn.Linear(self.input_size, 1)
        self.mu = nn.Parameter(th.Tensor(1))
        self.multi_labels = multi_labels
        self.reset_parameters()

        self.marks2 = nn.Linear(
            in_features=units_mlp[1], out_features=marks)
        self.mark_time = nn.Linear(
                in_features=self.input_size, out_features=units_mlp[1])
        self.mark_activation = self.get_mark_activation(mark_activation)

    def reset_parameters(self):
        nn.init.uniform_(self.w, b=0.001)


    def log_mark_pmf(
            self, 
            query_representations:th.Tensor, 
            history_representations:th.Tensor):
        
        p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.mark_time(history_representations))), dim=-1)
        
        p_m = p_m + epsilon(dtype=p_m.dtype, device=p_m.device)

        return th.log(p_m)

    
    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.LongTensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None,
            sampling: Optional[bool] = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        
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

        history_representations = take_3_by_2(                         
            representations, index=prev_times_idxs)                   # [B,T,D]

        self.mu.data = non_neg_param(self.mu.data)
        check_tensor(self.mu.data, positive=True)

        v_h_t = self.w_t(history_representations)                         #[B,T,1]
        v_h_t = v_h_t.squeeze(-1)                                         #[B,T]
        delta_t = query - prev_times
        w_delta_t = self.w * delta_t                     # [B,T]
    
        exp_term = th.clamp(w_delta_t + v_h_t, max=80) #Avoids infinity. 
        exp_term = th.exp(exp_term) 
        ground_intensity = self.mu + exp_term
        check_tensor(ground_intensity, positive=True)
        ground_intensity = ground_intensity + epsilon(dtype=ground_intensity.dtype, device=ground_intensity.device)
        log_ground_intensity = th.log(ground_intensity)
        check_tensor(log_ground_intensity)

        log_mark_pmf = self.log_mark_pmf(
                                query_representations=query_representations,
                                history_representations=history_representations
                            )
        check_tensor(log_mark_pmf)

        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask
        
        exp_1, exp_2 = v_h_t + w_delta_t, v_h_t                         # [B,T]
        exp_1, exp_2 = exp_1 * intensity_mask, exp_2 * intensity_mask   # [B,T]
        
        poisson_integral = self.mu * delta_t                            # [B,T]
                
        ground_intensity_integrals = th.exp(exp_1) - th.exp(exp_2)
        ground_intensity_integrals = ground_intensity_integrals / self.w                # [B,T]
        
        ground_intensity_integrals = ground_intensity_integrals + poisson_integral 
        
        check_tensor(ground_intensity_integrals * intensity_mask,
                     positive=True)
        
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
                artifacts)                      # [B,T,M], [B,T,M], [B,T], Dict

    def get_mark_activation(self, mark_activation):
        if mark_activation == 'relu':
            mark_activation = th.relu
        elif mark_activation == 'tanh':
            mark_activation = th.tanh
        elif mark_activation == 'sigmoid':
            mark_activation = th.sigmoid
        return mark_activation
    