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


class FNN(CumulativeDecoder):
    """The base FNN model. See https://proceedings.neurips.cc/paper_files/paper/2019/file/39e4973ba3321b80f37d9b55f63ed8b8-Paper.pdf. 
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
            name: Optional[str] = 'fnn',
            **kwargs):

        if constraint_mlp is None:
            print("Warning! MLP decoder is unconstrained. Setting to `nonneg`")
            constraint_mlp = "nonneg"

        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        input_size = units_mlp[0] - enc_size
        super(FNN, self).__init__(
            name=name,
            do_zero_subtraction=do_zero_subtraction,
            model_log_cm=model_log_cm,
            input_size=input_size,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mlp = MLP(
            units=units_mlp[1:],
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=units_mlp[0],
            activation_final=activation_final_mlp)
        self.mu = nn.Parameter(th.Tensor(self.marks))
        self.reset_parameters()

    def cum_intensity(
            self,
            query_representations:th.Tensor, 
            history_representations:th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the cumulative intensity and a mask
        """
        hidden = th.cat(
            [query_representations, history_representations],
            dim=-1)                                        # [B,T,units_mlp[0]]
        intensity_itg = self.mlp(hidden.float())                    # [B,T,output_size]

        return intensity_itg

    def diff_cum_marked_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None
            ):
        
        #query.requires_grad = True
        
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
        
        intensity_integrals_q = self.cum_intensity(
                                    query_representations=query_representations,
                                    history_representations=history_representations)

        delta_t = query - prev_times                                  # [B,T]
        delta_t = delta_t.unsqueeze(dim=-1)                           # [B,T,1]
        poisson_term = self.mu * delta_t                # [B,T,M]
        

        intensity_integrals_q = intensity_integrals_q + poisson_term

        # Remove masked values and add epsilon for stability
        intensity_integrals_q = \
            intensity_integrals_q * intensity_mask.unsqueeze(-1)

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
                        
            intensity_integrals_z = self.cum_intensity(
                                            query_representations=query_representations_z,
                                            history_representations=history_representations_z)

            intensity_integrals_z = intensity_integrals_z * intensity_mask_z.unsqueeze(-1)
             
            intensity_integrals_q = th.clamp(
                intensity_integrals_q - intensity_integrals_z, min=0.
            ) + intensity_integrals_z 
            intensity_integrals_q = intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=intensity_integrals_q.dtype,
                device=intensity_integrals_q.device) * query.unsqueeze(-1)
            
            intensity_integrals = intensity_integrals_q - intensity_integrals_z
            intensity_mask = intensity_mask * intensity_mask_z

        else:
            intensity_integrals_q = intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=intensity_integrals_q.dtype,
                device=intensity_integrals_q.device) * query.unsqueeze(-1)
            
            intensity_integrals = intensity_integrals_q

        check_tensor(intensity_integrals * intensity_mask.unsqueeze(-1),
                     positive=True)
        return intensity_integrals, intensity_mask

    def marked_intensity(
            self, 
            query:th.Tensor,
            intensity_integrals:th.Tensor
    ):
        assert(query.requires_grad)
        # Compute derivative of the integral
        grad_outputs = th.zeros_like(intensity_integrals, requires_grad=True)
        grad_inputs = th.autograd.grad(
            outputs=intensity_integrals,
            inputs=query,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True)[0]
        marked_intensity = th.autograd.grad(
            outputs=grad_inputs,
            inputs=grad_outputs,
            grad_outputs=th.ones_like(grad_inputs, requires_grad=True),
            retain_graph=True,
            create_graph=True)[0]
        return marked_intensity

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
        
        self.mu.data = non_neg_param(self.mu.data)
        check_tensor(self.mu.data, positive=True)
        
        query.requires_grad = True
        intensity_integrals, intensity_mask = self.diff_cum_marked_intensity(
                            events=events,
                            query=query,
                            prev_times=prev_times,
                            prev_times_idxs=prev_times_idxs,
                            pos_delta_mask=pos_delta_mask,
                            is_event=is_event,
                            representations=representations,
                            representations_mask=representations_mask
                    )
        marked_intensity = self.marked_intensity(
                                    query=query, 
                                    intensity_integrals=intensity_integrals
        )
        query.requires_grad = False

        check_tensor(marked_intensity, positive=True, strict=True)
        
        ground_intensity_integrals = th.sum(intensity_integrals, dim=-1)
        ground_intensity = th.sum(marked_intensity, dim=-1)
        log_ground_intensity = th.log(ground_intensity)

        mark_pmf = marked_intensity / ground_intensity.unsqueeze(-1)
        log_mark_pmf = th.log(mark_pmf)
        
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