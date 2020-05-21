import sys,os,argparse,time
import numpy as np
import torch
import utils
from datetime import datetime
import pickle
tstart=time.time()
import math

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',               default=0,                          type=int,     help='(default=%(default)d)')
parser.add_argument('--device',             default='cpu',                      type=str,     help='gpu id')
parser.add_argument('--experiment',         default='2_task_groups',            type =str,    help='Mnist or dissertation')
parser.add_argument('--approach',           default='PUGCL',                    type =str,    help='Method, always Lifelong Uncertainty-aware learning')
parser.add_argument('--data_path',          default='data/data.csv',            type=str,     help='gpu id')
parser.add_argument('--output',             default='',                         type=str,     help='')
parser.add_argument('--checkpoint_dir',     default='../checkpoints_2_tasks',   type=str,   help='')
parser.add_argument('--batch_size',         default=64,                         type=int,     help='')
parser.add_argument('--parameter',          default='',                         type=str,     help='')
parser.add_argument('--n_epochs',           default=100,                        type=int,     help='')
parser.add_argument('--lr',                 default=0.01,                       type=float,   help='')
parser.add_argument('--hidden_size',        default=800,                        type=int,     help='')

# Bayesian HYPER-PARAMETERS
parser.add_argument('--MC_samples',         default='10',           type=int,     help='Number of Monte Carlo samples')
parser.add_argument('--rho',                default='-3',           type=float,   help='Initial rho')
parser.add_argument('--sigma1',             default='0.0',          type=float,   help='STD foor the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sigma2',             default='6.0',          type=float,   help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi',                 default='0.25',         type=float,   help='weighting factor for prior')

parser.add_argument('--resume',             default='no',           type=str,     help='resume?')
parser.add_argument('--sti',                default=0,              type=int,     help='starting task?')

args=parser.parse_args()
utils.print_arguments(args)

# Arrays for paramater search:
n_epochs = [25, 50, 75, 100, 150, 200]
lr = [0.001, 0.005, 0.01, 0.03, 0.06, 0.1]
hidden_size = [120, 180, 240, 400, 600, 800]

# Set seed for stable results
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Check if Cuda is available
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Using device:", args.device)

# PUGCL with 1 task:
from data import dataloader_2_tasks as dataloader

# Import training approach:
from training_method import PUGCL

# Import bayesian model used:
from bayesian_model.bayesian_network2 import BayesianNetwork

print()
print("Starting this session on: ")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

# Load data:
print("Loading data...")
data, task_outputs, input_size = dataloader.get(data_path=args.data_path)
print("Input size =", input_size, "\nTask info =", task_outputs)
args.num_tasks = len(task_outputs)
args.input_size = input_size
args.task_outputs = task_outputs
pickle.dump(data, open( "data/data_2_task.p", "wb" ))

# Dict for best loss
best_loss = {}
best_loss['Loss'] = np.inf
best_loss['Model'] = None

for parameter_search in range(50):
    # Determine the parameters for this run:
    args.n_epochs = np.random.choice(n_epochs)
    args.lr = np.random.choice(lr)
    args.hidden_size = np.random.choice(hidden_size)
    task_order = np.arange(len(task_outputs))
    np.random.shuffle(task_order)

    print()
    print('-'*100)
    print("Starting testing for parameters: Epoch = " + str(args.n_epochs) + " Lr = " + str(args.lr) + " Hidden_size = " + str(args.hidden_size) + ' Task order: ' + str(task_order))
    print('-'*100)
    # Name experiment based on parameters:
    args.approach = 'PUGCL' + '_epochs_:' + str(args.n_epochs) + '_lr_:' + str(args.lr) + '_hidden_:' + str(args.hidden_size) + '_task_order_:' + str(task_order)

    # Checkpoint for this run
    checkpoint = utils.make_directories(args)
    args.checkpoint = checkpoint

    # Initialize Bayesian network
    #print("Initializing network...")
    model = BayesianNetwork(args).to(args.device)

    # Initialize Lul approach
    approach = PUGCL(model, args=args)


    # Iterate over the tasks:
    loss = np.zeros((len(task_outputs), len(task_outputs)), dtype=np.float32)
    task_count = 0
    for task, n_class in np.array(task_outputs)[task_order.astype(int)]:
        task_count += 1
        print('*'*100)
        print('Task {:2d} ({:s})'.format(task, data[task]['name']))
        print('*'*100)

        # Get data:
        xtrain = data[task]['train']['x'][:,1:].type(torch.float32).to(args.device)
        ytrain = data[task]['train']['y'].type(torch.float32).to(args.device)
        xvalid = data[task]['valid']['x'][:,1:].type(torch.float32).to(args.device)
        yvalid = data[task]['valid']['y'].type(torch.float32).to(args.device)

        # Start training
        print("Starting training for the tasks in group: ", task)
        approach.train(task, xtrain, ytrain, xvalid, yvalid)
        print('_'*100)

        # Validate for this task group:
        for u, n in np.array(task_outputs)[task_order[0:task_count].astype(int)]:
            xtest = data[u]['test']['x'][:,1:].type(torch.float32).to(args.device)
            ytest = data[u]['test']['y'].type(torch.float32).to(args.device)
            test_loss, test_error = approach.eval(u, xtest, ytest, debug=True)
            print("Test on task {:2d} - {:15s}: Loss={:.3f} Error={:.3f}".format(u, data[u]['name'], test_loss, test_error))
            loss[task, u] = test_loss

        # Save
        print("Saving at " + args.checkpoint)
        np.savetxt(os.path.join(args.checkpoint, '{}_{}_{}.txt'.format(args.experiment, args.approach, args.seed)), loss, '%.5f')

    # Check whether loss for this model is better than best loss:
    mean_loss = np.mean(loss[task,:])
    if math.isnan(mean_loss):
        continue
    if mean_loss < best_loss['Loss']:
        best_loss['Loss'] = mean_loss
        best_loss['Model'] = args.approach
    print(best_loss['Loss'])
    print(best_loss['Model'])
