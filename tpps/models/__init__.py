from argparse import Namespace
from torch import nn
from pprint import pprint

from tpps.models.base.enc_dec import EncDecProcess
from tpps.models.base.modular import ModularProcess
from tpps.models.poisson import PoissonProcess

from tpps.models.encoders.base.encoder import Encoder
from tpps.models.encoders.gru import GRUEncoder
from tpps.models.encoders.identity import IdentityEncoder
from tpps.models.encoders.constant import ConstantEncoder
from tpps.models.encoders.mlp_variable import MLPVariableEncoder
from tpps.models.encoders.stub import StubEncoder
from tpps.models.encoders.self_attention import SelfAttentionEncoder

from tpps.models.decoders.base.decoder import Decoder

from tpps.models.decoders.lnm import LNM
from tpps.models.decoders.lnm_p import LNM_P
from tpps.models.decoders.lnm_pp import LNM_PP

from tpps.models.decoders.lnm_joint import JOINT_LNM

from tpps.models.decoders.sthp import STHP
from tpps.models.decoders.sthp_pp import STHP_PP

from tpps.models.decoders.rmtpp import RMTPPDecoder
from tpps.models.decoders.rmtpp_p import RMTPP_P
from tpps.models.decoders.rmtpp_pp import RMTPP_PP

from tpps.models.decoders.fnn import FNN
from tpps.models.decoders.fnn_p import FNN_P
from tpps.models.decoders.fnn_pp import FNN_PP 
from tpps.models.decoders.fnn_d import FNN_D 
from tpps.models.decoders.fnn_dd import FNN_DD 

from tpps.models.decoders.thp import THP
from tpps.models.decoders.thp_p import THP_P
from tpps.models.decoders.thp_pp import THP_PP
from tpps.models.decoders.thp_d import THP_D
from tpps.models.decoders.thp_dd import THP_DD

from tpps.models.decoders.sahp import SAHP
from tpps.models.decoders.sahp_p import SAHP_P
from tpps.models.decoders.sahp_pp import SAHP_PP
from tpps.models.decoders.sahp_d import SAHP_D
from tpps.models.decoders.sahp_dd import SAHP_DD

ENCODER_CLASSES = {
    "gru": GRUEncoder,
    "identity": IdentityEncoder,
    "constant": ConstantEncoder,
    "mlp-variable": MLPVariableEncoder,
    "stub": StubEncoder,
    "selfattention": SelfAttentionEncoder
    }

DECODER_JOINT_CLASSES= {
    "mlp-cm": FNN,
    "thp": THP,
    "sahp":SAHP,
    "joint-lnm":JOINT_LNM
}


DECODER_DISJOINT_CLASSES = {
    "lnm": LNM,
    "lnm+":LNM_P,
    "lnm++":LNM_PP,
    
    "rmtpp": RMTPPDecoder,
    "rmtpp+": RMTPP_P,
    "rmtpp++": RMTPP_PP,
    
    "fnn": FNN,
    "fnn+": FNN_P,
    "fnn++": FNN_PP,
    "fnn-d": FNN_D,
    "fnn-dd": FNN_DD,

    "thp+": THP_P,
    "thp++": THP_PP,
    "thp-d": THP_D,
    "thp-dd": THP_DD,
    
    "sahp+":SAHP_P,
    "sahp++":SAHP_PP,
    "sahp-d":SAHP_D,
    "sahp-dd":SAHP_DD,

    "sthp":STHP,
    "sthp++":STHP_PP
    }


DECODER_CLASSES = {**DECODER_JOINT_CLASSES, **DECODER_DISJOINT_CLASSES}


ENCODER_NAMES = sorted(list(ENCODER_CLASSES.keys()))
DECODER_NAMES = sorted(list(DECODER_JOINT_CLASSES.keys()) + list(DECODER_DISJOINT_CLASSES.keys()))


CLASSES = {"encoder": ENCODER_CLASSES, "encoder_histtime": ENCODER_CLASSES, "encoder_histmark": ENCODER_CLASSES, 
           "decoder": DECODER_CLASSES}


NAMES = {"encoder": ENCODER_NAMES, "encoder_time": ENCODER_NAMES, "encoder_mark": ENCODER_NAMES,
          "decoder": DECODER_NAMES}


def instantiate_encoder_or_decoder(
        args: Namespace, component="encoder") -> nn.Module:
    assert component in {"encoder", "encoder_histtime", "encoder_histmark", "decoder"}
    prefix, classes = component + '_', CLASSES[component]
    if component in ["encoder_histtime", "encoder_histmark"]:
        prefix = 'encoder_'
        kwargs = {
            k[len(prefix):]: v for
            k, v in args.__dict__.items() if k.startswith(prefix)}
        if component == 'encoder_histtime':
            kwargs.update({"encoding":args.encoder_histtime_encoding})
        else:
            kwargs.update({"encoding":args.encoder_histmark_encoding})
    else:
        kwargs = {
            k[len(prefix):]: v for
            k, v in args.__dict__.items() if k.startswith(prefix)}
    
    kwargs["marks"] = args.marks 

    name = args.__dict__[component]
        
    if name not in classes:
        raise ValueError("Unknown {} class {}. Must be one of {}.".format(
            component, name, NAMES[component]))

    component_class = classes[name]
    component_instance = component_class(**kwargs) 

    return component_instance


def get_model(args: Namespace) -> EncDecProcess:
    args.decoder_units_mlp = args.decoder_units_mlp + [args.marks] 

    decoder: Decoder 
    decoder = instantiate_encoder_or_decoder(args, component="decoder")
    decoder_type = 'joint' if args.decoder in DECODER_JOINT_CLASSES else 'disjoint'

    if decoder.input_size is not None:
        args.encoder_units_mlp = args.encoder_units_mlp + [decoder.input_size]
    if args.encoder is not None:
        encoder: Encoder
        encoder = instantiate_encoder_or_decoder(args, component="encoder") 
        process = EncDecProcess(
        encoder=encoder, encoder_time=None, encoder_mark=None, decoder=decoder, multi_labels=args.multi_labels)
    else:
        encoder_time: Encoder
        encoder_mark: Encoder
        encoder_time = instantiate_encoder_or_decoder(args, component="encoder_histtime")
        encoder_mark = instantiate_encoder_or_decoder(args, component="encoder_histmark")
        process = EncDecProcess(
        encoder=None, encoder_time=encoder_time, encoder_mark=encoder_mark, 
        decoder=decoder, multi_labels=args.multi_labels, decoder_type=decoder_type)       
    if args.include_poisson: 
        processes = {process.name: process}
        processes.update({"poisson": PoissonProcess(marks=process.marks)})
        process = ModularProcess(
            processes=processes, args=args) 
    process = process.to(device=args.device)

    return process
