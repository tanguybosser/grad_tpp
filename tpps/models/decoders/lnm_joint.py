import math
import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder

from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor

class JOINT_LNM(VariableHistoryDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://arxiv.org/pdf/2210.15294.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            n_mixture: int,
            units_mlp: List[int],
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            embedding_constraint: Optional[str] = None,
            emb_dim: Optional[int] = 2,
            mark_activation: Optional[str] = 'relu',
            name: Optional[str] = 'joint-lnm',
            **kwargs):
        super(JOINT_LNM, self).__init__(
            name=name,
            input_size=units_mlp[0],
            marks=marks,
            encoding=encoding, 
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,  
            **kwargs)
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        self.marks = marks
        self.n_mixture = n_mixture
        self.mu = nn.Linear(in_features=units_mlp[0], out_features=marks * n_mixture)
        self.s = nn.Linear(in_features=units_mlp[0], out_features=marks * n_mixture)
        self.w = nn.Linear(in_features=units_mlp[0], out_features=marks * n_mixture)
        self.marks1 = nn.Linear(
            in_features=units_mlp[0], out_features=units_mlp[1])
        self.marks2 = nn.Linear(
            in_features=units_mlp[1], out_features=marks)
        self.mark_activation = self.get_mark_activation(mark_activation)

    
    def log_mark_pmf(
            self, 
            query_representations:th.Tensor, 
            history_representations:th.Tensor):
        
        p_m = th.softmax(
            self.marks2(
                self.mark_activation(self.marks1(history_representations))), dim=-1)
        
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
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
    
        query.requires_grad = True
        
        batch_size = query.shape[0]

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

        
        b,l,k,c = query.shape[0], query.shape[1], self.marks, self.n_mixture
        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)  # [B,T,D] 
        delta_t = query - prev_times  # [B,T]
        delta_t = delta_t.unsqueeze(-1).unsqueeze(-1)  # [B,T,1, 1]
        delta_t = th.relu(delta_t) 
        delta_t = delta_t + (delta_t == 0).float() * epsilon(
        dtype=delta_t.dtype, device=delta_t.device)
        delta_t = th.log(delta_t)
    
        mu = self.mu(history_representations)  # [B,T,K*C]
        mu = mu.view(b,l,k,c) #[B,T,K,C]
        std = th.exp(self.s(history_representations))
        std = std.view(b,l,k,c)        
        
        w = self.w(history_representations)
        w = w.view(b,l,k,c)
        w = th.softmax(w, dim=-1)
        
        log_mark_pmf = self.log_mark_pmf(
                            query_representations=query_representations,
                            history_representations=history_representations
        )
        
        mark_pmf = th.exp(log_mark_pmf)

        cum_f_k = w * 0.5 * (
                1 + th.erf((delta_t - mu) / (std * math.sqrt(2))))
        cum_f_k = th.clamp(th.sum(cum_f_k, dim=-1), max=1-1e-6) #[B,T,K]
        
        grad_outputs = th.zeros_like(cum_f_k, requires_grad=True)
        grad_inputs = th.autograd.grad(
            outputs=cum_f_k,
            inputs=query,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True)[0]
        f_k = th.autograd.grad(
            outputs=grad_inputs,
            inputs=grad_outputs,
            grad_outputs=th.ones_like(grad_inputs),
            retain_graph=True,
            create_graph=True)[0]
        query.requires_grad = False
        #conditional distribution
        f_k = f_k + epsilon(dtype=f_k.dtype, device=f_k.device) #[B,T,K]
        
        #Joint distribution
        f_joint = f_k * mark_pmf #[B,T,K] 
        
        #Marginal of times 
        f = th.sum(f_joint, dim=-1)

        #Joint CDF
        cum_f_joint = cum_f_k * mark_pmf  
        
        #Marginal CDF
        cum_f = th.sum(cum_f_joint, dim=-1) #[B,T]
        
        one_min_cum_f = 1. - cum_f
        one_min_cum_f = th.relu(one_min_cum_f) + epsilon(
            dtype=cum_f.dtype, device=cum_f.device)
        
        log_ground_intensity = th.log(f / one_min_cum_f)
        ground_intensity_integrals = - th.log(one_min_cum_f) + epsilon(eps=1e-7,
            dtype=cum_f.dtype, device=cum_f.device)
        
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask
        
        check_tensor(log_ground_intensity)
        check_tensor(log_mark_pmf)
        check_tensor(ground_intensity_integrals * intensity_mask, positive=True)


        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h = history_representations[th.arange(batch_size), last_event_idx,:]
        artifacts = {}
        artifacts['last_h'] = last_h.detach().cpu().numpy()
    
        return log_ground_intensity, log_mark_pmf, ground_intensity_integrals, intensity_mask, artifacts 

    def get_mark_activation(self, mark_activation):
        if mark_activation == 'relu':
            mark_activation = th.relu
        elif mark_activation == 'tanh':
            mark_activation = th.tanh
        elif mark_activation == 'sigmoid':
            mark_activation = th.sigmoid
        return mark_activation