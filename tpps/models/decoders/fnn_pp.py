import torch as th
import torch.nn as nn

from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.fnn_p import FNN_P
from tpps.models.base.process import Events

from tpps.utils.index import take_3_by_2
from tpps.utils.stability import epsilon, check_tensor
from tpps.utils.nnplus import non_neg_param


class FNN_PP(FNN_P):
    """A mlp decoder based on the cumulative approach.

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either none, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        mc_prop_est: Proportion of numbers of samples for the MC method,
                     compared to the size of the input. (Default=1.).
        do_zero_subtraction: If `True` the class computes
            Lambda(tau) = Lambda'(tau) - Lambda'(0)
            in order to enforce Lambda(0) = 0. Defaults to `True`.
        emb_dim: Size of the embeddings (default=2).
        encoding: Way to encode the events: either times_only, or temporal.
            Defaults to times_only.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            n_mixture: int, 
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
            mark_activation: Optional[str] = 'relu',
            hist_time_grouping: Optional[str] = 'summation',
            name: Optional[str] = 'fnn++',
            **kwargs):

        if constraint_mlp is None:
            print("Warning! MLP decoder is unconstrained. Setting to `nonneg`")
            constraint_mlp = "nonneg"

        super(FNN_PP, self).__init__(
            name=name,
            n_mixture=n_mixture,
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
            mark_activation=mark_activation,
            hist_time_grouping=hist_time_grouping,
            **kwargs)

   
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
        
        # Add grads for query to compute derivative
        query.requires_grad = True
        
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

        b,l = query.shape
        representations_time = representations[0:b,:,:]
        representations_mark = representations[b:,:,:]

        history_representations_time = take_3_by_2(                            
            representations_time, index=prev_times_idxs)                   # [B,T,D]
        history_representations_mark = take_3_by_2(                          
            representations_mark, index=prev_times_idxs)
        
        self.mu.data = non_neg_param(self.mu.data)
        check_tensor(self.mu.data, positive=True)
        
        ground_intensity_integrals_q = self.cum_ground_intensity(
                                    query_representations=query_representations,
                                    history_representations=history_representations_time)

        delta_t = query - prev_times                                  # [B,T]
        delta_t = delta_t.unsqueeze(dim=-1)                           # [B,T,1]
        poisson_term = self.mu * delta_t                # [B,T,M]
        

        ground_intensity_integrals_q = ground_intensity_integrals_q + poisson_term.squeeze(-1)

        # Remove masked values and add epsilon for stability
        ground_intensity_integrals_q = \
            ground_intensity_integrals_q * intensity_mask

        # Optional zero substraction
        if self.do_zero_subtraction: #Computes Lambda(0)
            
            query_representations_z, intensity_mask_z = self.get_query_representations(
                                        events=events,
                                        query=prev_times,
                                        prev_times=prev_times,
                                        prev_times_idxs=prev_times_idxs,
                                        pos_delta_mask=pos_delta_mask,
                                        is_event=is_event,
                                        representations=representations,
                                        representations_mask=representations_mask)  # [B,T,enc_size], [B,T]

                        
            ground_intensity_integrals_z = self.cum_ground_intensity(
                                            query_representations=query_representations_z,
                                            history_representations=history_representations_time)

            ground_intensity_integrals_z = ground_intensity_integrals_z * intensity_mask_z
             
            ground_intensity_integrals_q = th.clamp(
                ground_intensity_integrals_q - ground_intensity_integrals_z, min=0.
            ) + ground_intensity_integrals_z 
            
            ground_intensity_integrals_q = ground_intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=ground_intensity_integrals_q.dtype,
                device=ground_intensity_integrals_q.device) * query
            
            ground_intensity_integrals = ground_intensity_integrals_q - ground_intensity_integrals_z
            
            intensity_mask = intensity_mask * intensity_mask_z

        else:
            ground_intensity_integrals_q = ground_intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=ground_intensity_integrals_q.dtype,
                device=ground_intensity_integrals_q.device) * query
            
            ground_intensity_integrals = ground_intensity_integrals_q

        check_tensor(ground_intensity_integrals * intensity_mask,
                     positive=True)
        
        ground_intensity = th.autograd.grad(
            outputs=ground_intensity_integrals,
            inputs=query,
            grad_outputs=th.ones_like(ground_intensity_integrals),
            retain_graph=True,
            create_graph=True)[0]
        query.requires_grad = False

        check_tensor(ground_intensity * intensity_mask, positive=True)

        log_ground_intensity = th.log(ground_intensity)
        

        log_mark_pmf = self.log_mark_pmf(
                        query_representations=query_representations, 
                        history_representations=history_representations_mark)
        
        check_tensor(log_mark_pmf * intensity_mask.unsqueeze(-1))

        idx = th.arange(0,intensity_mask.shape[1]).to(intensity_mask.device)
        mask = intensity_mask * idx
        last_event_idx  = th.argmax(mask, 1)
        batch_size = query.shape[0]
        last_h_t = history_representations_time[th.arange(batch_size), last_event_idx,:]
        last_h_m = history_representations_mark[th.arange(batch_size), last_event_idx,:] #[B,D]
        artifacts = {}
        artifacts['last_h_t'] = last_h_t.detach().cpu().numpy()
        artifacts['last_h_m'] = last_h_m.detach().cpu().numpy()
            
        return (log_ground_intensity,
                log_mark_pmf,
                ground_intensity_integrals, 
                intensity_mask,
                artifacts)  # [B,T,M], [B,T,M], [B,T], Dict