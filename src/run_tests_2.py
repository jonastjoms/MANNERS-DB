import sys,os,argparse,time
import numpy as np
import torch
import utils
from datetime import datetime
import pickle

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',               default=0,              type=int,     help='(default=%(default)d)')
parser.add_argument('--device',             default='cpu',          type=str,     help='gpu id')
parser.add_argument('--experiment',         default='mnist2',       type =str,    help='Mnist or dissertation')
parser.add_argument('--approach',           default='lul',          type =str,    help='Method, always Lifelong Uncertainty-aware learning')
parser.add_argument('--data_path',          default='/Users/jonastjomsland/UCB/dissertation/src/data_cleaning/data.csv',     type=str,     help='gpu id')

# Training parameters
parser.add_argument('--output',             default='',             type=str,     help='')
parser.add_argument('--checkpoint_dir',     default='../checkpoints_10k/mnist2_lul',    type=str,   help='')
parser.add_argument('--n_epochs',           default=1,              type=int,     help='')
parser.add_argument('--batch_size',         default=64,             type=int,     help='')
parser.add_argument('--lr',                 default=0.01,           type=float,   help='')
parser.add_argument('--hidden_size',        default=600,           type=int,     help='')
parser.add_argument('--parameter',          default='',             type=str,     help='')

# UCB HYPER-PARAMETERS
parser.add_argument('--MC_samples',         default='10',           type=int,     help='Number of Monte Carlo samples')
parser.add_argument('--rho',                default='-3',           type=float,   help='Initial rho')
parser.add_argument('--sigma1',             default='0.0',          type=float,   help='STD foor the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sigma2',             default='6.0',          type=float,   help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi',                 default='0.25',         type=float,   help='weighting factor for prior')

parser.add_argument('--resume',             default='no',           type=str,     help='resume?')
parser.add_argument('--sti',                default=1,              type=int,     help='starting task?')

args=parser.parse_args()
utils.print_arguments(args)

# Set seed for stable results
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)

# Check if Cuda is available
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Using device:", args.device)

# Checkpoint
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint
print()

# MNIST with two tasks:
from data_cleaning import social_approp as dataloader

# Import Lifelong Uncertainty-aware Learning approach:
from bayesian_model.lul_2 import Lul

# Import model used:
from bayesian_model.bayesian_network_2 import BayesianNetwork

########################################################################################################################

# Load data:
print("Loading data...")
data, task_outputs, input_size = dataloader.get(data_path=args.data_path)
print("Input size =", input_size, "\nTask info =", task_outputs)
print("Number of data samples: ", len(data[0]['train']['x']))
args.num_tasks = len(task_outputs)
args.input_size = input_size
args.task_outputs = task_outputs
data = pickle.load(open( "structured_data/data.p", "rb" ))

# Initialize Bayesian network
print("Initializing network...")
model = BayesianNetwork(args).to(args.device)

# Initialize Lul approach
print("Initialize Lifelong Uncertainty-aware Learning")
approach = Lul(model, args=args)
print("-"*100)

# Load stored model:
checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device=args.device)

# Iterate over the two tasks:
# Circle:
xtest = data[1]['test']['x'].type(torch.float32).to(args.device)
ytest = data[1]['test']['y'].type(torch.float32).to(args.device)

# Get input and target
input, target, = xtest[1:2,1:].to(args.device), ytest[0].to(args.device)
# Get output
print("Scene number: ", int(xtest[1,0].detach().numpy()))
outputs = model(input, sample = False)
questions = ['Vacuum cleaning', 'Mopping the floor', 'Carry warm food', 'Carry cold food', 'Carry drinks', 'Carry small objects (plates, toys)', 'Carry big objects (tables, chairs)', 'Cleaning (Picking up stuff) / Starting conversation']
i = 0
for question in questions:
    print(question + ' ', outputs[0][0].detach().numpy()[i], np.exp(outputs[0][0].detach().numpy()[i+8]))
    i += 1
