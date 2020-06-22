# Predictivee Uncertainty Guided Continual Learning
# This work is an extension to Ebrahimi et al. (https://github.com/SaynaEbrahimi/UCB)
# Main differences are the addition of a custom loss function, and minor changes to the train() method
import os,sys,time
import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
from bayesian_model.bayesian_sgd import BayesianSGD


class UCB(object):

    def __init__(self, model, args, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=1000):
        self.model = model
        self.device = args.device
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.init_lr = args.lr
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.MC_samples = args.MC_samples

        self.output = args.output
        self.checkpoint = args.checkpoint
        self.num_tasks = args.num_tasks

        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)

    def train(self, task, xtrain, ytrain, xvalid, yvalid, loss_epochs):

        # Update learning rate based on parameter uncertainty:
        params_dict = self.update_lr(task)
        self.optimizer = BayesianSGD(params=params_dict)

        # Best loss:
        best_loss = np.inf

        # Store best model
        best_model = copy.deepcopy(self.model.state_dict())
        lr = self.init_lr

        patience = self.lr_patience

        # Iterate over number of epochs
        try:
            # Use a pickle file to dump validation loss per epoch, per task
            for epoch in range(self.n_epochs):
                # Train
                clock0=time.time()
                self.train_epoch(task, xtrain, ytrain)
                clock1=time.time()
                train_loss, train_error = self.eval(task, xtrain, ytrain)
                clock2=time.time()

                # Validation:
                valid_loss, valid_error = self.eval(task, xvalid, yvalid)
                loss_epochs['loss'][task].append(valid_loss.detach().numpy())
                loss_epochs['error'][task].append(valid_error.detach().numpy())
                # Print learning rate
                #print(' Learning rate: {:.3f} |'.format(lr), end='')
                # Print validation loss and error:
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Validation loss: {:.3f} |'.format(epoch+1,
                    1000*self.batch_size*(clock1-clock0)/xtrain.size(0),1000*self.batch_size*(clock2-clock1)/xtrain.size(0),
                    train_loss),end='')
                print(' Validation error: {:.3f} |'.format(train_error), end='')

                # Check if loss is nan
                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("\nLoss became nan, stopping training with saved model")
                    break

                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=copy.deepcopy(self.model.state_dict())
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience

                        params_dict = self.update_lr(task, adaptive_lr=True, lr=lr)
                        self.optimizer=BayesianSGD(params=params_dict)

                # Increase leanring rate if possible:
                if epoch > 30 and train_loss > 2:
                    lr = 0.08
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                print()

        except KeyboardInterrupt:
            print()

        # Restore best model:
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task)
        return loss_epochs

    def update_lr(self, task, adaptive_lr=False, lr=None):
        params_dict = []
        if task == 0:
            params_dict.append({'params': self.model.parameters(), 'lr': self.init_lr})
        else:
            # Iterate over layers in model
            for name in self.modules_names_without_cls:
                n = name.split('.')
                if len(n) == 1:
                    m = self.model._modules[n[0]]
                elif len(n) == 3:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
                elif len(n) == 4:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
                else:
                    print (name)
                if adaptive_lr is True:
                    params_dict.append({'params': m.weight_rho, 'lr': lr})
                    params_dict.append({'params': m.bias_rho, 'lr': lr})
                else:
                    # Uncertainty in weights and bias:
                    w_uncertainty = torch.log1p(torch.exp(m.weight_rho.data))
                    b_uncertainty = torch.log1p(torch.exp(m.bias_rho.data))
                    # Update learning rate based on weight uncertainty
                    params_dict.append({'params': m.weight_mu, 'lr': torch.mul(w_uncertainty, self.init_lr)})
                    params_dict.append({'params': m.bias_mu, 'lr': torch.mul(b_uncertainty,self.init_lr)})
                    params_dict.append({'params': m.weight_rho, 'lr':self.init_lr})
                    params_dict.append({'params': m.bias_rho, 'lr':self.init_lr})

        return params_dict

    def find_modules_names(self, with_classifier=False):
        modules_names = []
        for name, p in self.model.named_parameters():
            if with_classifier is False:
                if not name.startswith('classifier'):
                    n = name.split('.')[:-1]
                    modules_names.append('.'.join(n))
            else:
                n = name.split('.')[:-1]
                modules_names.append('.'.join(n))

        modules_names = set(modules_names)
        return modules_names

    def logs(self, task):
        log_prior = 0.0
        log_variational_posterior = 0.0
        for name in self.modules_names_without_cls:
            n = name.split('.')
            if len(n) == 1:
                m = self.model._modules[n[0]]
            elif len(n) == 3:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]

            log_prior += m.log_prior
            log_variational_posterior += m.log_variational_posterior

        log_prior += self.model.classifier[task].log_prior
        log_variational_posterior += self.model.classifier[task].log_variational_posterior

        return log_prior, log_variational_posterior

    def train_epoch(self, task, x, y):
        self.model.train()

        # Variable for shuffling
        index = np.arange(x.size(0))
        np.random.shuffle(index)
        index = torch.LongTensor(index).to(self.device)

        num_batches = len(x)//self.batch_size
        j = 0
        # Iterate over batches
        for i in range(0, len(index), self.batch_size):
            print("\rBatch: " + str(i)+"/"+str(len(index)), end=" ")
            if i + self.batch_size <= len(index):
                batch = index[i:i + self.batch_size]
            else:
                batch = index[i:]
            inputs, targets, = x[batch].to(self.device), y[batch].to(self.device)

            # Forward pass:
            loss, error = self.elbo_loss(inputs, targets, task, num_batches, sample=True)
            loss = loss.to(self.device)

            # Backward pass:
            #self.model.cuda()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #self.model.cuda()

            # Gradient step:
            self.optimizer.step()

    def eval(self, task, x, y, debug=False):
        total_loss = 0
        total_error = 0
        total_num = 0
        self.model.eval()

        index = np.arange(x.size(0))
        index = torch.as_tensor(index, device=self.device, dtype=torch.int64)

        with torch.no_grad():
            num_batches = len(x)//self.batch_size
            # Iterate over batches
            for i in range(0, len(index), self.batch_size):
                if i + self.batch_size <= len(index):
                    batch = index[i:i + self.batch_size]
                else:
                    batch = index[i:]
                inputs, targets, = x[batch].to(self.device), y[batch].to(self.device)

                # Forward pass:
                outputs = self.model(inputs, sample = False, sample_last_layer = False)
                output = outputs[task]
                loss, error = self.elbo_loss(inputs, targets, task, num_batches, sample=False, debug=debug)

                total_loss += loss.detach()*len(batch)
                total_error += error.detach()*len(batch)
                total_num += len(batch)

        return total_loss/total_num, total_error/total_num

    def set_model_(model, state_dict):
        model.model.load_state_dict(copy.deepcopy(state_dict))

    def loss(self, outputs, target, input):
        len_target = outputs.shape[1]//2
        # If no CL, split in 2 batches for circle and arrow actions:
        mask = input[:,0].detach().numpy()
        if (self.num_tasks == 1):
            no_cl  = True
            # Circle is mask non-zero elements
            circle_indices = np.nonzero(mask)
            # Mask is zero for rows with arrow
            arrow_indices = np.multiply(mask, np.arange(input.shape[0]))
            arrow_indices = np.where(arrow_indices == 0)[0]
            # Handle first element
            if mask[0] == 1:
                arrow_indices = arrow_indices[1:]
            # Mean is first 16 elements in output
            # First 8 is circle mean
            circle_mean = outputs[circle_indices[0],0:len_target//2]
            circle_target = target[circle_indices[0],:]
            # Second 8 is arrow mean
            arrow_mean = outputs[arrow_indices,len_target//2:len_target]
            arrow_target = target[arrow_indices,:]
            # Get log varience, last 16 elements
            # First 8 is circle variance
            log_variance_circle = outputs[circle_indices[0],len_target:-8]
            # Last 8 is arrow variance
            log_variance_arrow = outputs[arrow_indices,-8:]
            # Residual regression term
            arrow_error = arrow_target - arrow_mean
            circle_error = circle_target - circle_mean
            error = torch.zeros(outputs.shape[0], len_target//2)
            error[arrow_indices, :] = arrow_error
            error[circle_indices, :] = circle_error
            arrow_res_loss = torch.mean(0.5*torch.exp(-log_variance_arrow)*torch.square(arrow_error), axis = 1)
            circle_res_loss = torch.mean(0.5*torch.exp(-log_variance_circle)*torch.square(circle_error), axis = 1)
            res_loss = torch.zeros(outputs.shape[0])
            res_loss[arrow_indices] = arrow_res_loss
            res_loss[circle_indices] = circle_res_loss
            # Uncertainty loss
            arrow_unc_loss = torch.mean(0.5*torch.exp(log_variance_arrow), axis = 1)
            circle_unc_loss = torch.mean(0.5*torch.exp(log_variance_circle), axis = 1)
            unc_loss = torch.zeros(outputs.shape[0])
            unc_loss[arrow_indices] = arrow_unc_loss
            unc_loss[circle_indices] = circle_unc_loss
            # Combined loss:
            loss = res_loss + unc_loss
            # Get RMSE and Average over batch:
            error = error.square()
            loss = torch.mean(loss)
            error = torch.sqrt(error.mean(0)).mean()
            return loss, error

        # Get log varience
        log_variance = outputs[:,len_target:]
        # Mean term
        mean = outputs[:,0:len_target]
        # Reshape mean if 16-task model:
        if mean.shape[1] == 1:
            target = target.view(mean.shape[0],1)
        # Residual regression term
        error = target-mean
        res_loss = torch.mean(0.5*torch.exp(-log_variance)*torch.square(error), axis = 1)
        # Uncertainty loss
        unc_loss = torch.mean(0.5*torch.exp(log_variance), axis = 1)
        # Combined loss:
        loss = res_loss + unc_loss
        # Average over batch:
        loss = torch.mean(loss)
        error = error.square()
        error = torch.sqrt(error.mean(0)).mean()

        return loss, error

    def elbo_loss(self, input, target, task, num_batches, sample, debug=False):
        if sample:
            log_priors, log_variational_posteriors, predictions = [], [], []
            for i in range(self.MC_samples):
                predictions.append(self.model(input, sample=sample, sample_last_layer = sample)[task])
                log_prior, log_variational_posterior = self.logs(task)
                log_priors.append(log_prior)
                log_variational_posteriors.append(log_variational_posterior)

            # Coefficients, balancing loss:
            w1 = 1.e-3
            w2 = 1.e-3
            w3 = 5.e-2

            outputs = torch.stack(predictions, dim=0).to(self.device)
            log_var = w1*torch.as_tensor(log_variational_posteriors, device=self.device).mean()
            log_p = w2*torch.as_tensor(log_priors, device=self.device).mean()

            # This is where a custom loss function is implemented:
            loss, error = self.loss(outputs.mean(0), target, input)
            loss = w3*loss.to(device=self.device)
            error = error.to(device=self.device)

            return (log_var - log_p)/num_batches + loss, error

        else:
            predictions = []
            for i in range(self.MC_samples):
                predictions.append(self.model(input, sample=False, sample_last_layer = False)[task])

            outputs = torch.stack(predictions, dim=0).to(self.device)

            #negative_log_likelihood = w3*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device = self.device)
            loss, error = self.loss(outputs.mean(0), target, input)
            loss = loss.to(device=self.device)
            error = error.to(device=self.device)

            return loss, error

    def save_model(self, task):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task)))
