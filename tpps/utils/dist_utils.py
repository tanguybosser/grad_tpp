import torch
from torch.distributions import Categorical


def adjust_tensor(x, a=0.0, b=1.0, *, epsilon=1e-4):
    # We accept that, due to rounding errors, x is not in the interval up to epsilon
    mask = (a - epsilon <= x) & (x <= b + epsilon)
    assert mask.all(), (x[~mask], a, b)
    return x.clamp(a, b)


def adjust_unit_tensor(x, epsilon=1e-4):
    return adjust_tensor(x, a=0.0, b=1.0, epsilon=epsilon)


def icdf_from_cdf(
    one_minus_cdf,
    alpha,
    epsilon=1e-7,
    target_precision=1e-10,
    warn_precision=4e-3,
    low=None,
    high=None,
):
    """
    Compute the quantiles of a distribution using binary search, in a vectorized way.
    """

    alpha = adjust_unit_tensor(alpha)
    alpha = 1 - alpha
    # alpha, _ = torch.broadcast_tensors(alpha, torch.zeros(batch_shape))
    # Expand to the left and right until we are sure that the quantile is in the interval
    expansion_factor = 4
    if low is None:
        low = torch.full(alpha.shape, -1.0, dtype=alpha.dtype)
        while (mask := one_minus_cdf(low) > alpha + epsilon).any():
            low[mask] *= expansion_factor
    else:
        low = low.clone()
    if high is None:
        high = torch.full(alpha.shape, 1.0, dtype=alpha.dtype)
        while (mask := one_minus_cdf(high) < alpha - epsilon).any():
            high[mask] *= expansion_factor
    else:
        high = high.clone()
    low, high, b = torch.broadcast_tensors(low, high, torch.zeros(alpha.shape))
    #Check with previous conformal what were the dimensions. 
    assert one_minus_cdf(low).shape == alpha.shape
    # Binary search
    prev_precision = None
    for _ in range(100):
        # To avoid "UserWarning: Use of index_put_ on expanded tensors is deprecated".
        low = low.clone()
        high = high.clone()
        precision = (high - low).max()
        # Stop if we have enough precision
        if precision < target_precision:
            break
        # Stop if we can not improve the precision anymore
        if prev_precision is not None and precision >= prev_precision:
            break
        mid = (low + high) / 2
        mask = one_minus_cdf(mid) > alpha
        low[mask] = mid[mask]
        high[~mask] = mid[~mask]
        prev_precision = precision

    if precision > warn_precision:
        print(f'Imprecise quantile computation with precision {precision}')
    #print('low end', low[0][-30:])   
    return low


def icdf(model, alpha, past_events, low=None, high=None):
    last_observed_time = past_events.times_final.double()
    epsilon = 1e-7
    if low is None:
        low = last_observed_time + epsilon
        low = low.unsqueeze(-1)
    if high is None:
        high = last_observed_time + 100   # This works because the maximum inter-event time is 10
        high = high.unsqueeze(-1)

    def cdf(query):
        cdf_value, cdf_mask = model.cdf(query=query, events=past_events)
        return cdf_value

    return icdf_from_cdf(cdf, alpha, low=low, high=high)


def get_logz_samples(model, past_events, nb_samples=500):
    device = past_events.times.device
    batch_shape = past_events.times.shape[:1]

    # Sample quantile level
    alpha = torch.rand(batch_shape + (nb_samples,), device=device, dtype=torch.float64)
    alpha = alpha.sort(dim=1).values
    # self.sanity_check_cdf(past_events)
    # Sample time
    time_sample = icdf(model, alpha, past_events)
    # Sample label
    log_density, log_mark_density, y_pred_mask = model.log_density(query=time_sample, events=past_events)
    label_sample = Categorical(logits=log_mark_density).sample()
    # Sample Z
    logz_sample = torch.gather(log_density, 2, label_sample.unsqueeze(2)).squeeze(2)
    assert logz_sample.shape == batch_shape + (nb_samples,)
    return logz_sample
