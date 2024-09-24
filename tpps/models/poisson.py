from tpps.models.base.enc_dec import EncDecProcess
from tpps.models.encoders.stub import StubEncoder
from tpps.models.decoders.poisson import PoissonDecoder

from typing import Optional


class PoissonProcess(EncDecProcess):
    def __init__(
            self,
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            **kwargs):
        super(PoissonProcess, self).__init__(
            encoder=StubEncoder(marks=marks),
            encoder_time=None,
            encoder_mark=None,
            decoder=PoissonDecoder(marks=marks),
            multi_labels=multi_labels)
        self.mu = self.decoder.mu
