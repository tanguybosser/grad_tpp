import pdb

import torch as th

from typing import Dict, Optional, Tuple

# from tpp.processes.hawkes.r_terms import get_r_terms as get_r_terms
# from tpp.processes.hawkes.r_terms_recursive import get_r_terms
from tpps.processes.hawkes.r_terms_recursive_v import get_r_terms
from tpps.utils.events import Events
from tpps.utils.history_bst import get_prev_times, get_prev_times_sampling
from tpps.utils.index import take_3_by_2, take_2_by_2


def decoder_fast(
        events: Events,
        query: th.Tensor,
        prev_times: th.Tensor,
        is_event: th.Tensor,
        alpha: th.Tensor,
        beta: th.Tensor,
        mu: th.Tensor,
        marks: Optional[int] = 1,
        time_prediction = False, 
        sampling = False
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
    """Compute the intensities for each query time given event
    representations.

    Args:
        events: [B,L] Times and labels of events.
        query: [B,T] Times to evaluate the intensity function.
        prev_times: [B,T] Times of events directly preceding queries.
        is_event: [B,T] A mask indicating whether the time given by
            `prev_times_idxs` corresponds to an event or not (a 1 indicates
            an event and a 0 indicates a window boundary).
        alpha: [M,M] The alpha matrix for the Hawkes process.
        beta: [M,M] The beta matrix for the Hawkes process.
        mu: [M,M] The mu vector for the Hawkes process.
        marks: The number of marks for the process (default=1).

    Returns:
        log_intensity: [B,T,M] The intensities for each query time for
            each mark (class).
        intensity_integrals: [B,T,M] The integral of the intensity from
            the most recent event to the query time for each mark. #int from t_n to t_{n+1}
        intensities_mask: [B,T] Which intensities are valid for further
            computation based on e.g. sufficient history available.

    """
    (batch_size, t), seq_len = query.shape, events.times.shape[-1]
    # term_2 = sum_n alpha_m_n exp (-beta_m_n t-t_i_m) (R_m_n_i + 1)
    if time_prediction: #Necessary ? 
        ((prev_times, prev_times_idxs),
         is_event, mask) = get_prev_times_sampling(
            query=query, events=events, allow_window=True)
    else:
        ((prev_times, prev_times_idxs),
        is_event, mask) = get_prev_times(
            query=query, events=events, allow_window=True, sampling=sampling)                  # [B,T]
    events_labels = th.argmax(events.labels, dim=-1).long()             # [B,L] #returns the actual mark at each pos of the seq.
    if time_prediction or sampling:
        repeat = int(query.shape[0]/events_labels.shape[0])
        events_labels = events_labels.repeat_interleave(repeat, dim=0)
        prev_labels = take_2_by_2(
            events_labels, index=prev_times_idxs - 1)
        r_terms = get_r_terms(events=events, beta=beta)  # [B,L,M,N]
        r_terms = r_terms.repeat_interleave(repeat, dim=0)
        window_r_term = th.zeros(
            size=(batch_size, 1, marks, marks),
            dtype=r_terms.dtype,
            device=r_terms.device)
        r_terms = th.cat([window_r_term, r_terms], dim=1)  # [B,L+1,M,N]
    else:
        prev_labels = take_2_by_2(
            events_labels, index=prev_times_idxs - 1)                       # [B,T] #for each elem in seq, return previous label.

        r_terms = get_r_terms(events=events, beta=beta)               # [B,L,M,N]
        window_r_term = th.zeros(
            size=(batch_size, 1, marks, marks),
            dtype=r_terms.dtype,
            device=r_terms.device)
        r_terms = th.cat([window_r_term, r_terms], dim=1)             # [B,L+1,M,N]
    r_terms_query = r_terms.reshape(
        batch_size, seq_len + 1, -1)                              # [B,L+1,M*N]
    r_terms_query = take_3_by_2(
        r_terms_query, index=prev_times_idxs)                     # [B,T,M*N]
    r_terms_query = r_terms_query.reshape(
        batch_size, t, marks, marks)                              # [B,T,M,N]

    delta_t = query - prev_times                                  # [B,T]

    # Compute exp( -beta_mn * (t - t_-) )
    arg = delta_t.unsqueeze(dim=-1).unsqueeze(dim=-1)             # [B,T,1,1]
    arg = - beta.reshape(1, 1, marks, marks) * arg                # [B,T,M,N]
    exp_mask = is_event * mask                                    # [B,T]
    exp_mask = exp_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)       # [B,T,1,1]
    arg = arg * exp_mask                                          # [B,T,M,N]
    exp = th.exp(arg)                                             # [B,T,M,N]
    exp_mask = is_event.unsqueeze(dim=-1).unsqueeze(dim=-1)       # [B,T,1,1]
    exp = exp * exp_mask                                          # [B,T,M,N]

    mark_range = th.arange(
        end=marks, device=events_labels.device, dtype=events_labels.dtype)
    ones = (prev_labels.unsqueeze(dim=-1) ==
            mark_range.unsqueeze(dim=0).unsqueeze(dim=0))    # [B,T,M]
    ones = ones.unsqueeze(dim=2).repeat(1, 1, marks, 1)  # [B,T,M,N]
    ones = ones.type(exp.dtype)

    r_terms_plus_one = r_terms_query + ones

    exp_intensity = exp * r_terms_plus_one
    exp_intensity = alpha.reshape(1, 1, marks, marks) * exp_intensity
    exp_intensity = th.sum(exp_intensity, dim=-1) #sum over all n (interaction terms)

    intensity = mu.reshape(1, 1, marks) + exp_intensity          # [B,T,M]

    log_intensity = th.log(intensity)                            # [B,T,M]

    intensity_integral = 1 - exp                                 # [B,T,M,N]
    intensity_integral = intensity_integral * exp_mask           # [B,T,M,N]
    intensity_integral = alpha.reshape(
        1, 1, marks, marks) * intensity_integral                 # [B,T,M,N]
    intensity_integral = intensity_integral / beta.reshape(
        1, 1, marks, marks)                                      # [B,T,M,N]
    intensity_integral = intensity_integral * r_terms_plus_one   # [B,T,M,N]
    intensity_integral = th.sum(intensity_integral, dim=-1)      # [B,T,M]

    # term_4 = mu (t - t_i)               ti < t
    term_4 = delta_t.unsqueeze(dim=-1) * mu.reshape(1, 1, marks)  # [B,T,M]

    intensity_integral = intensity_integral + term_4              # [B,T,M]

    intensities_mask = events.within_window(query, time_prediction=time_prediction, sampling=sampling)               # [B,T]
    return log_intensity, intensity_integral, intensities_mask, dict()
