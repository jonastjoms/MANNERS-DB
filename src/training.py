import sys,os,argparse,time
import numpy as np
import torch
import utils
from datetime import datetime
import pickle
tstart = time.time()


# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',               default=0,                          type=int,     help='(default=%(default)d)')
parser.add_argument('--device',             default='cpu',                      type=str,     help='gpu id')
parser.add_argument('--n_tasks',            default= 1,                         type =int,    help='1, 2 or 16, referring to tasks')
parser.add_argument('--approach',           default='UCB',                      type =str,    help='Method, always Lifelong Uncertainty-aware learning')
parser.add_argument('--test_data_path',     default='data/data_test.csv',       type=str,     help='gpu id')
parser.add_argument('--training_data_path', default='data/data_train.csv',      type=str,     help='gpu id')
parser.add_argument('--output',             default='',                         type=str,     help='')
parser.add_argument('--checkpoint_dir',     default='../checkpoints',           type=str,     help='')
parser.add_argument('--batch_size',         default=64,                         type=int,     help='')
parser.add_argument('--parameter',          default='',                         type=str,     help='')
parser.add_argument('--n_epochs',           default=200,                        type=int,     help='')
parser.add_argument('--lr',                 default=0.06,                       type=float,   help='')
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

# Set seed for stable results
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)

# Check if Cuda is available
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Using device:", args.device)

# Set n_task specific variables
if args.n_tasks == 1:
    # Load data:
    from data import dataloader_1_tasks as train_dataloader
    from data import dataloader_1_tasks_test as test_dataloader
elif args.n_tasks == 2:
    # Load data:
    from data import dataloader_2_tasks as train_dataloader
    from data import dataloader_2_tasks_test as test_dataloader
else:
    # Load data:
    from data import dataloader_16_tasks as train_dataloader
    from data import dataloader_16_tasks_test as test_dataloader

# Import training approach:
from UCB_modified import UCB

# Import bayesian model used:
from bayesian_model.bayesian_network import BayesianNetwork

print()
print("Starting this session on: ")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

# Load data:
print("Loading data...")
data_train, task_outputs, input_size = train_dataloader.get(data_path=args.training_data_path)
data_test = test_dataloader.get(data_path=args.test_data_path)
print("Input size =", input_size, "\nTask info =", task_outputs)
args.num_tasks = len(task_outputs)
args.input_size = input_size
args.task_outputs = task_outputs
# Dump data so train/test split is stored
pickle.dump(data_train, open( "data/train_data_1_task.p", "wb" ))
pickle.dump(data_test, open( "data/test_data_1_task.p", "wb" ))

# Add to experiment name if testing:
#args.approach += '_test_'

# Checkpoint for this run
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint

# Initialize Bayesian network
model = BayesianNetwork(args).to(args.device)

# Initialize UCB
approach = UCB(model, args=args)

# Array to store test loss
loss = np.zeros((len(task_outputs), len(task_outputs)), dtype=np.float32)
# Dict to store validation loss per epoch for each task
loss_epochs = {}
loss_epochs['loss'] = {}
loss_epochs['error'] = {}
# Iterate over the tasks:
for task, n_class in task_outputs[args.sti:]:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(task, data_train[task]['name']))
    print('*'*100)

    # Store validation loss per epoch
    loss_epochs['loss'][task] = []
    loss_epochs['error'][task] = []
    # Get data:
    xtrain = data_train[task]['train']['x'][:,1:].type(torch.float32).to(args.device)
    ytrain = data_train[task]['train']['y'].type(torch.float32).to(args.device)
    xvalid = data_train[task]['valid']['x'][:,1:].type(torch.float32).to(args.device)
    yvalid = data_train[task]['valid']['y'].type(torch.float32).to(args.device)

    # Start training
    print("Starting training for the tasks in group: ", task)
    loss_epochs = approach.train(task, xtrain, ytrain, xvalid, yvalid, loss_epochs)
    print('_'*100)

    # Test for all task groups:
    for u in range(args.n_tasks):
        xtest = data_test[u]['test']['x'][:,1:].type(torch.float32).to(args.device)
        ytest = data_test[u]['test']['y'].type(torch.float32).to(args.device)
        test_loss, test_error = approach.eval(u, xtest, ytest, debug=True)
        print("Test on task {:2d} - {:15s}: Loss={:.3f} Error={:.3f}".format(u, data_test[u]['name'], test_loss, test_error))
        loss[task, u] = test_loss

    # Save
    print("Saving at " + args.checkpoint)
    np.savetxt(os.path.join(args.checkpoint, '{}_{}_{}.txt'.format(args.n_tasks, args.approach, args.seed)), loss, '%.5f')
pickle.dump(loss_epochs, open(os.path.join(args.checkpoint, 'loss_epoch_dict.p'), "wb" ))
pickle.dump(loss, open(os.path.join(args.checkpoint, 'loss_task_dict.p'), "wb" ))
