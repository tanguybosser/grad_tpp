import torch as th
import torch.nn as nn

from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.base.cumulative import CumulativeDecoder
from tpps.models.base.process import Events

from tpps.pytorch.models import MLP

from tpps.utils.encoding import encoding_size
from tpps.utils.index import take_3_by_2
from tpps.utils.stability import epsilon, check_tensor
from tpps.utils.nnplus import non_neg_param


class FNN_P(CumulativeDecoder):
    """
    The FNN+ model. 
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
            name:Optional[str] = 'fnn+', 
            **kwargs):

        if constraint_mlp is None:
            print("Warning! MLP decoder is unconstrained. Setting to `nonneg`")
            constraint_mlp = "nonneg"

        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        input_size = units_mlp[0] - enc_size
        super(FNN_P, self).__init__(
            name=name,
            do_zero_subtraction=do_zero_subtraction,
            model_log_cm=model_log_cm,
            input_size=input_size,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        units = units_mlp[1:-1] + [n_mixture]
        self.mlp = MLP(
            units=units,
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=units_mlp[0],
            activation_final=activation_final_mlp)
        
        self.mu = nn.Parameter(th.Tensor(1))
        self.reset_parameters()
        
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


    def get_mark_activation(self, mark_activation):
        if mark_activation == 'relu':
            mark_activation = th.relu
        elif mark_activation == 'tanh':
            mark_activation = th.tanh
        elif mark_activation == 'sigmoid':
            mark_activation = th.sigmoid
        return mark_activation
    

    def cum_ground_intensity(
            self,
            query_representations:th.Tensor, 
            history_representations:th.Tensor
    ) -> th.Tensor:
        """Compute the cumulative ground intensity.
        """
        
        hidden = th.cat(
            [query_representations, history_representations],
            dim=-1)                                        # [B,T,units_mlp[0]]
        output = self.mlp(hidden.float())                    # [B,T,output_size]
        ground_intensity_integral = th.sum(output, dim=-1)
        
        return ground_intensity_integral

    
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

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D] 
        
        self.mu.data = non_neg_param(self.mu.data)
        check_tensor(self.mu.data, positive=True)
        
        ground_intensity_integrals_q = self.cum_ground_intensity(
                                    query_representations=query_representations,
                                    history_representations=history_representations)

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

            history_representations_z = take_3_by_2(representations, index=prev_times_idxs)                   # [B,T,D] 
                        
            ground_intensity_integrals_z = self.cum_ground_intensity(
                                            query_representations=query_representations_z,
                                            history_representations=history_representations_z)

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
                        history_representations=history_representations)
        
        check_tensor(log_mark_pmf * intensity_mask.unsqueeze(-1))

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