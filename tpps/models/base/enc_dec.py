import torch as th

from typing import Dict, Optional, Tuple

from tpps.models.decoders.base.decoder import Decoder
from tpps.models.encoders.base.encoder import Encoder
from tpps.models.base.process import Process
from tpps.utils.events import Events
from tpps.utils.history_bst import get_prev_times
from tpps.utils.logical import xor
from tpps.utils.stability import epsilon, check_tensor
from tpps.utils.dist_utils import icdf_from_cdf

class EncDecProcess(Process):
    """A parametric encoder decoder process.

    Args
        encoder: The encoder.
        decoder: The decoder.

    """
    def __init__(self,
                 encoder: Encoder,
                 encoder_time:Encoder, 
                 encoder_mark:Encoder,
                 decoder: Decoder,
                 multi_labels: Optional[bool] = False,
                 decoder_type: Optional[str] = 'joint',
                 **kwargs):
        if encoder is not None:
            assert encoder.marks == decoder.marks
            name = '_'.join([encoder.name, decoder.name])
            marks = encoder.marks
            self.enc_dec_hidden_size = encoder.output_size
            if decoder.input_size is not None:
                assert encoder.output_size == decoder.input_size
        elif encoder_time is not None:
            assert encoder_time.marks == decoder.marks
            assert encoder_mark.marks == decoder.marks
            name = '_'.join([encoder_time.name, encoder_mark.name, decoder.name])
            marks = encoder_time.marks
            if decoder.input_size is not None:
                assert encoder_time.output_size == decoder.input_size
                assert encoder_mark.output_size == decoder.input_size
            self.enc_dec_hidden_size = encoder_time.output_size
        else:
            name = kwargs.pop("name")
            marks = kwargs.pop("marks")
        super(EncDecProcess, self).__init__(name=name, marks=marks, **kwargs)
        self.encoder = encoder
        self.encoder_time = encoder_time
        self.encoder_mark = encoder_mark
        self.decoder = decoder
        self.multi_labels = multi_labels
        self.decoder_type = decoder_type

    def intensity(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the ground intensity at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            intensity: [B,T,M] The intensities for each query time and for each
                mark (class).
            intensity_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        log_intensity, _, intensity_mask, _ = self.artifacts(
            query=query, events=events)
        return th.exp(log_intensity), intensity_mask

    def log_density(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the log joint densities at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            log_density: [B,T,M] The log joint densities for each query time and for each
                mark (class).
            log_mark_density: [B,T,M] The log mark PMFs for each query time and for each
                mark (class).
            density_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        log_ground_intensity, log_mark_density, ground_intensity_integral ,intensity_mask, _ = self.artifacts(
            query=query, events=events)
        
        log_ground_intensity = log_ground_intensity * intensity_mask
        log_mark_density = log_mark_density * intensity_mask.unsqueeze(-1)
        ground_intensity_integral = ground_intensity_integral * intensity_mask

        ground_intensity = th.exp(log_ground_intensity)
        ground_density = ground_intensity * th.exp(-ground_intensity_integral)

        mark_density = th.exp(log_mark_density)        
        joint_density = ground_density.unsqueeze(-1) * mark_density
        joint_density = joint_density + epsilon(dtype=joint_density.dtype)
        log_joint_density = th.log(joint_density)
        
        check_tensor(log_joint_density)
        check_tensor(log_mark_density)

        return log_joint_density, log_mark_density, intensity_mask

    def neg_log_likelihood(
            self, events: Events, test=False, time_scaling: Optional[float] = None
            ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the negative log likelihood of events.

        Args:
            events: [B,L] Times and labels of events.
            test: If True, computes the CDF at query times.
        Returns:
            loss_t: [B] The time loss.
            loss_m: [B] The mark loss.
            loss_w: [B] The window loss.
            loss_mask: [B] Which losses are valid for further
                computation based on e.g. at least one element in sequence has
                a contribution.
            artifacts: Other useful items, e.g. the relevant window of the
                sequence.

        """
        artifacts = {}
        events_times = events.get_times(postpend_window=True)         # [B,L+1] 
        
        log_ground_intensity, log_mark_density, ground_intensity_integral, intensity_mask, artifacts_modular = self.artifacts(
                query=events_times, events=events)  # [B,L+1], [B,L+1,M], [B,L+1]
        
        window_intensity_mask = intensity_mask[:, -1]          # [B]
        intensity_mask = intensity_mask[:, :-1]                # [B,L]

        log_ground_intensity = log_ground_intensity[:,:-1] #[B,L]
        log_ground_intensity = log_ground_intensity * intensity_mask
        
        labels = events.labels                                        # [B,L,M]
        log_mark_density = log_mark_density[:,:-1,:] #[B,L,M]
        true_log_mark_density = log_mark_density * labels 
        true_log_mark_density = th.sum(true_log_mark_density, dim=-1)   
        true_log_mark_density = true_log_mark_density * intensity_mask

        window_integral = ground_intensity_integral[:,-1]  #[B,1]
        window_integral = window_integral * window_intensity_mask
        
        ground_intensity_integral = ground_intensity_integral[:,:-1]        #[B,L]
        ground_intensity_integral = ground_intensity_integral * intensity_mask

        add_window_integral = 1 - events.final_event_on_window.type(
            intensity_mask.dtype)                                     # [B]
        window_integral = window_integral * add_window_integral       # [B]

        loss_t = -th.sum(log_ground_intensity, dim=-1) + th.sum(ground_intensity_integral, dim=-1)
        loss_m = -th.sum(true_log_mark_density, dim=-1)
        loss_w = window_integral
        artifacts['loss_t'] = loss_t
        artifacts['loss_m'] = loss_m 
        artifacts['loss_w'] = loss_w       
        if time_scaling is not None:
            loss_t = 1/(time_scaling) * loss_t 

        loss_mask = th.sum(intensity_mask, dim=-1)                    # [B]
        loss_mask = (loss_mask > 0.).type(intensity_mask.dtype)                    # [B]
        defined_window_integral = window_intensity_mask * add_window_integral
        no_window_integral = 1 - add_window_integral                    # [B]
        window_mask = xor(defined_window_integral, no_window_integral)  # [B]
        loss_mask = loss_mask * window_mask
        #CDF computation    
        if test:
            cumulative_density = 1 - th.exp(-ground_intensity_integral) #[B, L]
            cumulative_density[~intensity_mask.bool()] = -1      #[B,L]
            artifacts["cumulative density"] = cumulative_density.detach()  
        
        if 'last_h' in artifacts_modular:
            artifacts['last_h'] = artifacts_modular['last_h']
        if 'last_h_t' in artifacts_modular:
            artifacts['last_h_t'] = artifacts_modular['last_h_t']
            artifacts['last_h_m'] = artifacts_modular['last_h_m']

        return loss_t, loss_m, loss_w, loss_mask, artifacts 
    
    def artifacts(
            self, query: th.Tensor, events: Events, prev_times: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the (log) ground intensities, log mark PMF, and ground intensity integrals at query times
        given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.
            prev_times: [B,T] Arrival time preceeding each event.
        Returns:
            log_ground_intensity: [B,T] The log ground intensities for each query time.
            log_mark_density: [B,T,M] The log ground PMF of marks for each query time.
            ground_intensity_integrals: [B,T] The integral of the ground intensity from
                the most recent event to the query time.
            intensities_mask: [B,T,M] Which intensities (PMF, integrals) are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        if self.encoder is not None:
            representations, representations_mask, artifacts = self.encode(
                events=events, encoding_type='encoder')                            # [B,L+1,D] [B,L+1], Dict
        elif self.encoder_time is not None:
            representations_time, representations_mask, artifacts = self.encode(
                events=events, encoding_type='encoder_time')
            representations_mark, representations_mask, artifacts = self.encode(
                events=events, encoding_type='encoder_mark')
            representations = th.cat((representations_time, representations_mark), 0) #[2B, L+1, D] [B, L+1]
        if prev_times is None:
            prev_times, is_event, pos_delta_mask = get_prev_times( #
                                                        query=query,
                                                        events=events,
                                                        allow_window=True)# ([B,T],[B,T]), [B,T], [B,T]
            prev_times, prev_times_idxs = prev_times  # [B,T], [B,T]
        else: 
            prev_times, is_event, pos_delta_mask, prev_times_idxs = prev_times
        return self.decode(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            is_event=is_event,
            pos_delta_mask=pos_delta_mask,
            representations=representations,
            representations_mask=representations_mask,
            artifacts=artifacts)

    def encode(self, events: Events, encoding_type: str) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the history representations.

        Args:
            events: [B,L] Times and labels of events.
            encoding_type: Which type of history must be returned (global, mark, or time). 
        Returns:
            representations: [B,L+1,H] The history representations for window start and each event.
            histories_mask: [B,L+1] A mask indicating which representations are well-defined.  
            artifacts: A dictionnary of whatever else you want to return. 
        """
        if encoding_type == "encoder":
            return self.encoder(events=events) 
        elif encoding_type == "encoder_time":
            return self.encoder_time(events=events)
        else:
            return self.encoder_mark(events=events)

    def decode(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            is_event: th.Tensor,
            pos_delta_mask: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the intensities for each query time given event
        representations.

        Args:
            events: [B,L] Times and labels of events.
            query: [B,T] Times to evaluate the intensity function.
            prev_times: [B,T] Times of events directly preceding queries.
            prev_times_idxs: [B,T] Indexes of times of events directly
                preceding queries. These indexes are of window-prepended
                events.
            pos_delta_mask: [B,T] A mask indicating if the time difference
                `query - prev_times` is strictly positive.
            is_event: [B,T] A mask indicating whether the time given by
                `prev_times_idxs` corresponds to an event or not (a 1 indicates
                an event and a 0 indicates a window boundary).
            representations: [B,L+1,D] Representations of window start and
                each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.
        Returns:
            log_ground_intensity: [B,T] The log ground intensities for each query times.
            log_mark_density: [B,T,M] The log mark PMFs for each query times.
            ground_intensity_integrals: [B,T] The integral of the ground intensity from
                the most recent event to the query time.
            intensities_mask: [B,T] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        return self.decoder(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            is_event=is_event,
            pos_delta_mask=pos_delta_mask,
            representations=representations,
            representations_mask=representations_mask,
            artifacts=artifacts)

    def cdf(
            self, query: th.Tensor, events: Events, prev_times: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the cdf at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.
            prev_times: [B,T] Arrival time preceeding each event.
        Returns:
            cdf: [B,T] The (ground) cdf evaluated at each query time.
            intensity_mask: [B,T] Which cdf are valid for further
                computation based on e.g. sufficient history available.

        """
        _, _, ground_intensity_integral, intensity_mask, _ = self.artifacts(
            query=query, events=events, prev_times=prev_times)
        cdf = 1 - th.exp(-ground_intensity_integral)
        return cdf, intensity_mask
    

    def one_minus_cdf(
            self, query: th.Tensor, events: Events, prev_times: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the 1-cdf at query times given events. More numerically stable than cdf.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.
            prev_times: [B,T] Arrival time preceeding each event.
        Returns:
            one_minus_cdf: [B,T] The 1 - (ground) cdf evaluated at each query time.
            intensity_mask: [B,T] Which cdf are valid for further
                computation based on e.g. sufficient history available.

        """
        _ ,_, ground_intensity_integral, intensity_mask, _ = self.artifacts(
            query=query, events=events, prev_times=prev_times)
        one_minus_cdf = th.exp(-ground_intensity_integral)
        return one_minus_cdf, intensity_mask
    
    def icdf(self, alpha, past_events, low=None, high=None, all_events=False, prev_times=None):
        #Alpha View must occur before function call 
        times = past_events.get_times() #[B,L]
        epsilon = 1e-6
        if low is None:
            low = times + epsilon
        if high is None:
            #high = last_observed_time + 100   # This works because the maximum inter-event time is 10
            high = times + 10
        def one_minus_cdf(query):
            one_minus_cdf_value, cdf_mask = self.one_minus_cdf(query=query, events=past_events, prev_times=prev_times)
            return one_minus_cdf_value
        times = icdf_from_cdf(one_minus_cdf, alpha, low=low, high=high)
        return times