from sys import int_info
import numpy as np

from argparse import ArgumentParser
from distutils.util import strtobool

from tpps.models import ENCODER_NAMES, DECODER_NAMES


def str_to_none(arg):
     return None if arg == 'None' else arg


def parse_args(allow_unknown=False):
    parser = ArgumentParser(allow_abbrev=False) 
    # Run configuration
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--padding-id", type=float, default=-1.,
                        help="The value used in the temporal sequences to "
                             "indicate a non-event.")
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--verbose', action='store_true',
                        help='If `True`, prints all the things.')
    parser.add_argument("--marks", type=int, default=None,
                        help="Numbe of marks.")
    # Common model hyperparameters
    parser.add_argument("--include-poisson",
                        type=lambda x: bool(strtobool(x)), default=True,
                        help="Include base intensity (where appropriate).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="The batch size to use for parametric model"
                             " training and evaluation.")
    parser.add_argument("--train-epochs", type=int, default=501,
                        help="The number of training epochs.")
    parser.add_argument("--use-coefficients",
                        type=lambda x: bool(strtobool(x)), default=True,
                        help="If true, the modular process will be trained "
                             "with trainable coefficients")
    parser.add_argument("--coefficients",
                         type=lambda x: None if x == 'None' else float(x), default=[], action='append',
                        help="Fix coefficients with which to train the modular process, which will remain fixed during training. "
                        "Only if use-coefficient is True. First value is for the model, second is for the Poisson process "
                         "Defaults to None, in which case coefficients are trained starting at [1, -2].")
    parser.add_argument("--multi-labels",    
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="Whether the likelihood is computed on "
                             "multi-labels events or not")
    parser.add_argument("--time-scale", type=float, default=1.,
                        help='Time scale used to prevent overflow')
    # Learning rate and patience parameters
    parser.add_argument("--lr-rate-init", type=float, default=0.001,
                        help='initial learning rate for optimization')
    parser.add_argument("--lr-poisson-rate-init", type=float, default=0.001,
                        help='initial poisson learning rate for optimization')
    parser.add_argument("--lr-scheduler",
                        choices=['plateau', 'step', 'milestones', 'cos',
                                 'findlr', 'noam', 'clr', 'calr'],
                        default='plateau',
                        help='method to adjust learning rate')
    parser.add_argument("--lr-scheduler-patience", type=int, default=1000,
                        help='lr scheduler plateau: Number of epochs with no '
                             'improvement after which learning rate will be '
                             'reduced')
    parser.add_argument("--lr-scheduler-step-size", type=int, default=10,
                        help='lr scheduler step: number of epochs of '
                             'learning rate decay.')
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.5,
                        help='learning rate is multiplied by the gamma to '
                             'decrease it')
    parser.add_argument("--lr-scheduler-warmup", type=int, default=10,
                        help='The number of epochs to linearly increase the '
                             'learning rate. (noam only)')
    parser.add_argument("--patience", type=int, default=20,
                        help="The patience for early stopping.")
    parser.add_argument("--loss-relative-tolerance", type=float,
                        default=0.0005,
                        help="The relative factor that the loss needs to "
                             "decrease by in order to not contribute to "
                             "patience. If `None`, will not use numerical "
                             "convergence to control early stopping. Defaults "
                             "to `None`.")
    parser.add_argument("--mu-cheat",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="If True, the starting mu value will be the "
                             "actual mu value. Defaults to False.")
    # Encoder specific hyperparameters
    parser.add_argument("--encoder", type=str, default=None,
                        choices=ENCODER_NAMES,
                        help="The type of encoder to use.")
    parser.add_argument("--encoder-histtime", type=str, default=None,
                        choices=ENCODER_NAMES,
                        help="The type of encoder to use to parametrize the density.")
    parser.add_argument("--encoder-histmark", type=str, default=None,
                        choices=ENCODER_NAMES,
                        help="The type of encoder to use to parametize the pmf.")
    # Encoder - Fixed history
    parser.add_argument("--encoder-history-size", type=int, default=3,
                        help="The (fixed) history length to use for fixed "
                             "history size parametric models.")
    # Encoder - Variable history
    parser.add_argument("--encoder-emb-dim", type=int, default=4,
                        help="Size of the embeddings. This is the size of the "
                             "temporal encoding and/or the label embedding if "
                             "either is used.")
    parser.add_argument("--encoder-encoding", type=str, default=None,
                        choices=["times_only", "marks_only", "concatenate",
                                 "temporal", "learnable",
                                 "temporal_with_labels", 
                                 "learnable_with_labels",
                                 "log_times_only",
                                 "log_concatenate"],
                        help="Type of the event encoding.")
    parser.add_argument("--encoder-histtime-encoding", type=str, default=None,
                        choices=["times_only", "marks_only", "concatenate",
                                 "temporal", "learnable",
                                 "temporal_with_labels", 
                                 "learnable_with_labels",
                                 "log_times_only",
                                 "log_concatenate"],
                        help="Type of the event encoding for the time encoder.")
    parser.add_argument("--encoder-histmark-encoding", type=str, default=None,
                        choices=["times_only", "marks_only", "concatenate",
                                 "temporal", "learnable",
                                 "temporal_with_labels", 
                                 "learnable_with_labels",
                                 "log_times_only",
                                 "log_concatenate"],
                        help="Type of the event encoding for the mark encoder.")
    parser.add_argument("--encoder-time-encoding", type=str,
                        default="relative", choices=["absolute", "relative"])
    parser.add_argument("--encoder-temporal-scaling", type=float, default=1.,
                        help="Rescale of times when using temporal encoding")
    parser.add_argument("--encoder-embedding-constraint", type=str,
                        default=None,
                        help="Constraint on the embeddings. Either `None`, "
                             "'nonneg', 'sigmoid', 'softplus'. "
                             "Defaults to `None`.")
    # Encoder - MLP
    parser.add_argument("--encoder-units-mlp", type=int, default=[],
                        nargs="+", metavar='N',
                        help="Size of hidden layers in the encoder MLP. "
                             "This will have the decoder input size appended "
                             "to it during model build.")
    parser.add_argument("--encoder-dropout-mlp", type=float, default=0.,
                        help="Dropout rate of the MLP")
    parser.add_argument("--encoder-activation-mlp", type=str, default="relu",
                        help="Activation function of the MLP")
    parser.add_argument("--encoder-constraint-mlp", type=str, default=None,
                        help="Constraint on the mlp weights. Either `None`, "
                             "'nonneg', 'sigmoid', 'softplus'. "
                             "Defaults to `None`.")
    parser.add_argument("--encoder-activation-final-mlp",
                        type=str, default=None,
                        help="Final activation function of the MLP.")
    # Encoder - RNN/Transformer
    parser.add_argument("--encoder-attn-activation",
                        type=str, default="softmax",
                        choices=["identity", "sigmoid", "softmax"],
                        help="Activation function of the attention "
                             "coefficients")
    parser.add_argument("--encoder-dropout-rnn", type=float, default=0.,
                        help="Dropout rate of the RNN.")
    parser.add_argument("--encoder-layers-rnn", type=int, default=1,
                        help="Number of layers for RNN and self-attention "
                             "encoder.")
    parser.add_argument("--encoder-units-rnn", type=int, default=32,
                        help="Hidden size for RNN and self attention encoder.")
    parser.add_argument("--encoder-n-heads", type=int, default=1,
                        help="Number of heads for the transformer")
    parser.add_argument("--encoder-constraint-rnn", type=str, default=None,
                        help="Constraint on the rnn/sa weights. Either `None`,"
                             "'nonneg', 'sigmoid', 'softplus'. "
                             "Defaults to `None`.")
    # Decoder specific hyperparameters
    parser.add_argument("--decoder", type=str, default="hawkes",
                        choices=DECODER_NAMES,
                        help="The type of decoder to use.")
    parser.add_argument("--decoder-mc-prop-est", type=float, default=1.,
                        help="Proportion of MC samples, "
                             "compared to dataset size")
    parser.add_argument("--decoder-model-log-cm",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="Whether the cumulative models the log integral"
                             "or the integral")
    parser.add_argument("--decoder-do-zero-subtraction",
                        type=lambda x: bool(strtobool(x)), default=True,
                        help="For cumulative estimation. If `True` the class "
                             "computes Lambda(tau) = f(tau) - f(0) "
                             "in order to enforce Lambda(0) = 0. Defaults to "
                             "`true`, where instead Lambda(tau) = f(tau).")
    # Decoder - Variable history
    parser.add_argument("--decoder-emb-dim", type=int, default=4,
                        help="Size of the embeddings. This is the size of the "
                             "temporal encoding and/or the label embedding if "
                             "either is used.")
    parser.add_argument("--decoder-encoding", type=str, default="times_only",
                        choices=["times_only", "marks_only", "concatenate",
                                 "temporal", "learnable",
                                 "temporal_with_labels",
                                 "learnable_with_labels", 
                                 "log_times_only", 
                                 "log_concatenate"],
                        help="Type of the event decoding.")
    parser.add_argument("--decoder-time-encoding", type=str,
                        default="relative", choices=["absolute", "relative"])
    parser.add_argument("--decoder-temporal-scaling", type=float, default=1.,
                        help="Rescale of times when using temporal encoding")
    parser.add_argument("--decoder-embedding-constraint", type=str,
                        default=None,
                        help="Constraint on the embeddings. Either `None`, "
                             "'nonneg', 'sigmoid', 'softplus'. "
                             "Defaults to `None`.")
    # Decoder - MLP
    parser.add_argument("--decoder-units-mlp", type=int ,default=[],
                        action='append',
                        help="Size of hidden layers in the decoder MLP. "
                             "This will have the number of marks appended "
                             "to it during model build.")
    parser.add_argument("--decoder-dropout-mlp", type=float, default=0.,
                        help="Dropout rate of the MLP")
    parser.add_argument("--decoder-activation-mlp", type=str, default="relu",
                        help="Activation function of the MLP")
    parser.add_argument("--decoder-activation-final-mlp",
                        type=str, default=None,
                        help="Final activation function of the MLP.")
    parser.add_argument("--decoder-constraint-mlp", type=str, default=None,
                        help="Constraint on the mlp weights. Either `None`, "
                             "'nonneg', 'sigmoid', 'softplus'. "
                             "Defaults to `None`.")
    # Decoder - RNN/Transformer
    parser.add_argument("--decoder-attn-activation", type=str,
                        default="softmax",
                        choices=["identity", "sigmoid", "softmax"],
                        help="Activation function of the attention "
                             "coefficients")
    parser.add_argument("--decoder-activation-rnn", type=str, default="relu",
                        help="Activation for the rnn.")
    parser.add_argument("--decoder-dropout-rnn", type=float, default=0.,
                        help="Dropout rate of the RNN")
    parser.add_argument("--decoder-layers-rnn", type=int, default=1,
                        help="Number of layers for self attention decoder.")
    parser.add_argument("--decoder-units-rnn", type=int, default=32,
                        help="Hidden size for self attention decoder.")
    parser.add_argument("--decoder-n-heads", type=int, default=1,
                        help="Number of heads for the transformer")
    parser.add_argument("--decoder-constraint-rnn", type=str, default=None,
                        help="Constraint on the rnn/sa weights. Either `None`,"
                             "'nonneg', 'sigmoid', 'softplus'. "
                             "Defaults to `None`.")
    # Decoder - MM
    parser.add_argument("--decoder-n-mixture", type=int, default=32,
                        help="Number of mixtures for the log normal mixture"
                             "model")
    parser.add_argument("--decoder-mark-activation", type=str, default='relu',
                        choices=['relu', 'tanh', 'sigmoid'],
                        help="Inner activation of the mark distribution. Either 'relu', 'tanh', 'sigmoid'")
    parser.add_argument("--decoder-hist-time-grouping", type=str, default='concatenation',
                        choices=['summation', 'concatenation'],
                        help="Grouping of history and event time embeddings in the computation of the mark pmf.")

    # Load and save
    parser.add_argument("--load-from-dir", type=str, default=None,
                        help="If not None, load data from a directory")
    parser.add_argument("--save-model-freq", type=int, default=25,
                        help="The best model is saved every nth epoch")
    parser.add_argument("--eval-metrics",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="The model is evaluated using several metrics")
    parser.add_argument("--eval-metrics-per-class",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="The model is evaluated using several metrics "
                             "per class")
    parser.add_argument("--save-results-dir", type=str, default=None,
                        help='Directory to save training/validation/test results')
    parser.add_argument("--dataset", type=str, default=None,
                        help='the dataset ow which to train the model')
    parser.add_argument("--split", type=int, default=None, 
                         help= 'Split on whihc to train the model')
    parser.add_argument("--load-check-file", 
                         type=str_to_none, default=None,
                         help='Load model from checkpoint')
    parser.add_argument("--save-check-dir", 
                         type=str_to_none, default=None,
                         help='Directory to save model checkpoints')
    parser.add_argument("--exp-name", 
                         type=str_to_none, default=None,
                         help='Additional name for the experiment. Defaults to None.')
    parser.add_argument("--use-softmax", 
                         type=lambda x: bool(strtobool(x)), default=True,
                         help='If True, the intensity coefficients are passed through a Softmax layer to scale in [0,1]. Defaults to True.')
    #Training
    parser.add_argument("--separate-training", 
                         type=lambda x: bool(strtobool(x)), default=False,
                         help='If True, trains the NLL-T and NLL-M terms separately.')
    parser.add_argument("--include-window", 
                         type=lambda x: bool(strtobool(x)), default=True,
                         help='Include window during training')
    parser.add_argument("--nll-scaling", 
                         type=float, default=None,
                         help='Scaling to apply to the time component of the NLL.')
    #Eval
    parser.add_argument("--model-name", 
                         type=str, default=None,
                         help='Name of the model to load for evaluation')
    if allow_unknown:
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()

    return args
