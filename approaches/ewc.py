import sys, time, os
import numpy as np
import torch
from copy import deepcopy
import approaches.utils as utils    
sys.path.append('..')
from approaches.arguments import get_args   
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import yaml
import pandas as pd
args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """
    def __init__(
        self, model, lamb=1, nepochs=100, sbatch=256, lr=0.001, lr_min=1e-6,
        lr_factor=1, lr_patience=5, clipgrad=1., drop_last_batch=False,
        linear_probing_epochs=0, clipfisher=False,
        args=None, log_name = None,
        srefsample=256, srefbatch=128,
        lamb_act=1, lamb_act_decay=False, lamb_act_decay_rate=0.1, lamb_act_decay_step=5
        ):

        self.model=model
        self.model_old=model
        self.fisher=None
        self.clipfisher = clipfisher

        self.nepochs = nepochs
        self.linear_probing_epochs = linear_probing_epochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min * 1/3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.drop_last_batch = drop_last_batch
        self.srefsample = srefsample
        self.srefbatch = srefbatch

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.lamb=lamb
        self.lamb_act = lamb_act
        self.lamb_act_decay = lamb_act_decay
        self.lamb_act_init = lamb_act
        self.lamb_act_decay_rate = lamb_act_decay_rate
        self.lamb_act_decay_step = lamb_act_decay_step
        #self.alpha = 0.9 # 1 == standard A-GEM
        #self.soft_agem = True
        self.tau = 0 # 0.1 # 0 == standard A-GEM
        print(f' lamb is {self.lamb}')   
        print(f'optim is {args.optim}') 

        self.log_name = log_name

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        return optimizer
    

    def register_dummies_into_buffer(self):
        for n, p in self.model.named_parameters():
            self.model.register_buffer('{}_fisher'.format(n.replace('.', '_')), torch.zeros_like(p)) 

    def load_from_buffers(self):
        self.fisher = {}
        for n, p in self.model.named_parameters():
            self.fisher[n] = getattr(self.model, '{}_fisher'.format(n.replace('.', '_'))) 
            
        
    def load_model(self, state_dict):   
        self.register_dummies_into_buffer()
        self.model.load_state_dict(state_dict)
        self.load_from_buffers()
        self.model_old = deepcopy(self.model)
        utils.freeze_model(self.model_old) # Freeze the weights

    def train(
            self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla,
            log_accs=False, defend=False, distill_folder=None, agem=False, reg_project=False,
            reg_turn_off_on_projection=True, act_reg=False,
            log_loss=False, freeze_bn=False, set_bn_eval=False
        ):

        if self.clipfisher and self.fisher is not None:
            FISHER_MAX = 1.0 / (self.lr * self.lamb)
            for n in self.fisher:
                self.fisher[n] = torch.clamp(self.fisher[n], max=FISHER_MAX)
        
        best_loss = np.inf
        #best_acc = 0
        best_model = utils.get_model(self.model)
        if freeze_bn:
            # Freeze BatchNorm layers
            self.model.apply(utils.freeze_bn_params)
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)

        if defend or log_loss:
            # Load inverted samples
            distill_names = [f for f in os.listdir(distill_folder) if f.endswith('.npz')]
            distill_names.sort()
            xref, yref, tref = [], [], []
            for t_id, distill_name in enumerate(distill_names):
                if t_id >= t:
                    break
                distilldata = np.load(os.path.join(distill_folder, distill_name))
                x_inv = torch.from_numpy(distilldata['x_dst'])
                y_inv = torch.from_numpy(distilldata['y_dst'])
                tinv = torch.full((x_inv.size(0),), t_id, dtype=torch.long, device=device)
                xref.append(x_inv)
                yref.append(y_inv)
                tref.append(tinv)
            self.xref = torch.cat(xref).to(device)
            self.yref = torch.cat(yref).to(device)
            self.tref = torch.cat(tref).to(device)
            self.ref_perm_log = torch.randperm(self.xref.size(0))
            # Initialize projection log list
            self.log_proj = []

        if (defend and act_reg) or log_loss:
            # Store reference activations
            self.model.eval()
            self.stored_ref_activations = []
            with torch.no_grad():
                for i in range(0, self.xref.size(0), self.srefbatch):
                    xb = self.xref[i:i+self.srefbatch].to(device)
                    feats = self.model.forward_no_head(xb)
                    self.stored_ref_activations.append(feats)
            self.stored_ref_activations = torch.cat(self.stored_ref_activations, dim=0).detach()
        
        if log_loss:
            # Initialize reference loss log list
            self.log_loss = []
        
        if log_accs:
            # Initialize accuracies log list
            self.log_accs = []

        # Loop epochs
        for e in range(self.nepochs):
            # Linear probing phase
            if e == 0 and self.linear_probing_epochs > 0:
                for name, param in self.model.named_parameters():
                    if f'heads.{t}' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                self.optimizer = self._get_optimizer(lr)
            # Full fine-tuning phase
            if e == self.linear_probing_epochs and self.linear_probing_epochs > 0:
                for param in self.model.parameters():
                    param.requires_grad = True
                if freeze_bn:
                    # Freeze BatchNorm layers
                    self.model.apply(utils.freeze_bn_params)
                lr /= self.lr_factor
                self.optimizer = self._get_optimizer(lr)
            # Train
            # clock0=time.time()
            # num_batch = xtrain.size(0)

            if act_reg and self.lamb_act_decay:
                self.lamb_act = self.lamb_act_init * (self.lamb_act_decay_rate ** (e // self.lamb_act_decay_step))
                print(f'Epoch {e}, lamb_act: {self.lamb_act:.4f}')

            self.train_epoch(
                t, xtrain, ytrain, e, data, log_accs, defend, agem, reg_project,
                reg_turn_off_on_projection, act_reg, log_loss, freeze_bn,
                set_bn_eval
            )

            # clock1=time.time()
            # train_loss,train_acc=self.eval(t,xtrain,ytrain)
            # clock2=time.time()
            # print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
            #     e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
            #     1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss, valid_acc=self.eval(t,xvalid,yvalid)
            print('task: {} epoch {} Valid: loss={:.3f}, acc={:5.1f}% |'.format(t, e, valid_loss,100*valid_acc),end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))

            # if valid_acc > best_acc:
            #     best_acc = valid_acc
            #     best_model = utils.get_model(self.model)
            # Adapt lr
            # if valid_loss < best_loss:
            #     best_loss = valid_loss
            #     best_model = utils.get_model(self.model)
            #     patience = self.lr_patience
            #     print(' *', end='')

            # else:
            #     patience -= 1
            #     if patience <= 0:
            #         lr /= self.lr_factor
            #         print(' lr={:.1e}'.format(lr), end='')
            #         if lr < self.lr_min:
            #             print()
            #         patience = self.lr_patience
            #         self.optimizer = self._get_optimizer(lr)
            print()
            best_model = utils.get_model(self.model)

        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.train()
        utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self.criterion, drop_last_batch=self.drop_last_batch)
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option

        for p_idx, (n, p ) in enumerate(self.model.named_parameters()):            
            self.model.register_buffer('{}_fisher'.format(n.replace('.', '_')), self.fisher[n])   

        # Save defense configuration
        if args.output_dir is not None:
            defense_config = {
                'defend': defend,
                'srefsample': self.srefsample,
                'srefbatch': self.srefbatch,
                'reg_project': reg_project,
                'reg_turn_off_on_projection': reg_turn_off_on_projection,
                'act_reg': act_reg,
                'lamb_act': self.lamb_act,
                'lamb_act_decay': self.lamb_act_decay,
                'lamb_act_decay_rate': self.lamb_act_decay_rate,
                'lamb_act_decay_step': self.lamb_act_decay_step,
                'linear_probing_epochs': self.linear_probing_epochs,
            }
            defense_config_file = os.path.join(args.output_dir, 'defense_config.yaml')
            with open(defense_config_file, 'w') as f:
                yaml.dump(defense_config, f, default_flow_style=False)

        if defend:
            # Save projections log
            if args.output_dir is not None:
                proj_log_file = os.path.join(args.output_dir, f'task{t}_AGEM_projections.csv')
            else:
                proj_log_file = self.log_name + f'_defend_{defend}_srefsample_{self.srefsample}_regproj_{reg_project}_regoff{reg_turn_off_on_projection}_task{t}_AGEM_projections.csv'
            pd.DataFrame(self.log_proj).to_csv(proj_log_file, index=False)

        if log_loss:
            # Save reference loss log
            if args.output_dir is not None:
                ref_loss_log_file = os.path.join(args.output_dir, f'task{t}_loss.csv')
            else:
                ref_loss_log_file = self.log_name + f'_defend_{defend}_srefsample_{self.srefsample}_regproj_{reg_project}_regoff{reg_turn_off_on_projection}_task{t}_loss.csv'
            pd.DataFrame(self.log_loss).to_csv(ref_loss_log_file, index=False)

        if log_accs:
            # Save accuracies log
            if args.output_dir is not None:
                acc_log_file = os.path.join(args.output_dir, f'task{t}_accuracy_progress.csv')
            else:
                acc_log_file = self.log_name + f'task{t}_accuracy_progress.csv'
            pd.DataFrame(self.log_accs).to_csv(acc_log_file, index=False)

        return
    
    def compute_reference_loss(self):
        idx = self.ref_perm_log
        if self.srefsample < self.xref.size(0):
            idx = idx[:self.srefsample]
        xref_batch = self.xref[idx].to(device)
        yref_batch = self.yref[idx].to(device)
        tref_batch = self.tref[idx].to(device)

        loss_ref_total = 0
        unique_tasks = tref_batch.unique()

        for task_id in unique_tasks:
            mask = tref_batch == task_id
            xref_task = xref_batch[mask]
            yref_task = yref_batch[mask]
            for i in range(0, xref_task.size(0), self.srefbatch):
                xref_sub = xref_task[i:i+self.srefbatch]
                yref_sub = yref_task[i:i+self.srefbatch]
                outputs_sub = self.model.forward(xref_sub)[task_id.item()]
                loss_ref_total += self.ce(outputs_sub, yref_sub)

        loss_ref_total /= len(unique_tasks)

        return loss_ref_total
    
    def compute_activation_loss(self):
        idx = self.ref_perm_log
        if self.srefsample < self.xref.size(0):
            idx = idx[:self.srefsample]
        xref_batch = self.xref[idx].to(device)
        stored_acts_batch = self.stored_ref_activations[idx].to(device)

        features_list = []
        for j in range(0, xref_batch.size(0), self.srefbatch):
            xref_small = xref_batch[j:j+self.srefbatch]
            feats = self.model.forward_no_head(xref_small)
            features_list.append(feats)
        features = torch.cat(features_list, dim=0)

        activation_loss = F.mse_loss(features, stored_acts_batch)

        return activation_loss

    def defend_step(
            self, agem, reg_project, reg_turn_off_on_projection, act_reg,
            freeze_bn, set_bn_eval
        ):
        # Store current gradient
        grad_cur = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_cur.append(param.grad.view(-1))
            else:
                grad_cur.append(torch.zeros(param.numel(), device=device))
        grad_cur = torch.cat(grad_cur)
        projected = False # Initialize projection flag

        # Sample reference data
        idx = torch.randperm(self.xref.size(0))
        if self.srefsample < self.xref.size(0):
            idx = idx[:self.srefsample]
        xref_batch = self.xref[idx].to(device)
        yref_batch = self.yref[idx].to(device)
        tref_batch = self.tref[idx].to(device)
        if act_reg:
            stored_acts_batch = self.stored_ref_activations[idx].to(device)

        self.model.eval()

        if agem:
            # Compute reference gradients
            self.optimizer.zero_grad()
            loss_ref_total = 0
            unique_tasks = tref_batch.unique()

            for task_id in unique_tasks:
                mask = (tref_batch == task_id)
                xref_task = xref_batch[mask]
                yref_task = yref_batch[mask]
                for j in range(0, xref_task.size(0), self.srefbatch):
                    xref_sub = xref_task[j:j+self.srefbatch]
                    yref_sub = yref_task[j:j+self.srefbatch]
                    outputs_sub = self.model.forward(xref_sub)[task_id.item()]
                    loss_ref_total += self.ce(outputs_sub, yref_sub)

            loss_ref_total /= len(unique_tasks)
            loss_ref_total.backward()

            # Store reference gradients
            grads_ref = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads_ref.append(param.grad.view(-1))
                else:
                    grads_ref.append(torch.zeros(param.numel(), device=device))
            grads_ref = torch.cat(grads_ref).detach()

            assert (grad_cur.shape == grads_ref.shape)

            # Project gradient (A-GEM)
            dotg = torch.dot(grad_cur, grads_ref)
            if dotg < 0: #-self.tau:
                projected = True
                grad_cur = grad_cur - (dotg / torch.dot(grads_ref, grads_ref)) * grads_ref
                # grad_proj = grad_cur - (dotg / torch.dot(grads_ref, grads_ref)) * grads_ref
                # if self.soft_agem:
                #     grad_cur = (1 - self.alpha) * grad_cur + self.alpha * grad_proj
                # else:
                #     grad_cur = grad_proj

        if act_reg:
            # Compute current reference activations
            self.optimizer.zero_grad()
            features_list = []
            for j in range(0, xref_batch.size(0), self.srefbatch):
                xref_small = xref_batch[j:j+self.srefbatch]
                feats = self.model.forward_no_head(xref_small)
                features_list.append(feats)
            features = torch.cat(features_list, dim=0)

            # Store activations gradients
            activation_loss = F.mse_loss(features, stored_acts_batch)
            (activation_loss * self.lamb_act).backward()
            grads_act = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads_act.append(param.grad.view(-1))
                else:
                    grads_act.append(torch.zeros(param.numel(), device=device))
            grads_act = torch.cat(grads_act).detach()
        
        # Set current task gradient (possibly projected)
        index = 0
        for param in self.model.parameters():
            n_param = param.numel()
            if param.grad is not None:
                param.grad.copy_(grad_cur[index:index + n_param].view_as(param))
            index += n_param

        if agem:
            if not (reg_project or (reg_turn_off_on_projection and projected)):
                # Add weight regularization term gradient
                for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                    if param.grad is not None:
                        reg_grad = self.lamb * self.fisher[name] * (param - param_old)
                        param.grad += reg_grad

        if act_reg:
            # Add activation gradient
            index = 0
            for param in self.model.parameters():
                n_param = param.numel()
                if param.grad is not None:
                    param.grad += grads_act[index:index + n_param].view_as(param)
                index += n_param

        self.model.train()
        if freeze_bn or set_bn_eval:
            # Set batch normalization to eval mode
            self.model.apply(utils.set_bn_eval)

        return projected

    def train_epoch(
            self, t, x, y, epoch, data, log_accs, defend, agem, reg_project,
            reg_turn_off_on_projection, act_reg, log_loss, freeze_bn,
            set_bn_eval
        ):
        
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)

        linear_probing = epoch < self.linear_probing_epochs
        if linear_probing or freeze_bn or set_bn_eval:
            # Set batch normalization to eval mode
            self.model.apply(utils.set_bn_eval)

        num_projections = 0

        if log_loss and epoch == 0:
            # Compute reference loss
            with torch.no_grad():
                self.model.eval()
                ref_loss = self.compute_reference_loss()
                act_loss = self.compute_activation_loss()
                self.model.train()
                if linear_probing or freeze_bn or set_bn_eval:
                    # Set batch normalization to eval mode
                    self.model.apply(utils.set_bn_eval)
            self.log_loss.append({
                'epoch': epoch,
                'batch': -1,
                'ref_loss': ref_loss.item(),
                'projection': False,
                'act_loss': act_loss.item()
            })

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r):
                b=r[i:i+self.sbatch]
            else:
                if self.drop_last_batch:
                    continue
                b=r[i:]
            images=x[b].to(device) 
            targets=y[b].to(device)

            self.optimizer.zero_grad()
            # Forward current model
            outputs = self.model.forward(images)[t]
            # Compute loss
            apply_reg = not ((agem and not reg_project) or linear_probing)
            loss = self.criterion(t, outputs, targets, apply_reg=apply_reg)
            # Backward
            loss.backward()

            projected = False
            if defend and not linear_probing:
                # Apply defense step
                projected = self.defend_step(
                    agem, reg_project, reg_turn_off_on_projection, act_reg,
                    freeze_bn, set_bn_eval
                )
                if projected:
                    num_projections += 1

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            if log_loss:
                # Compute reference loss
                with torch.no_grad():
                    self.model.eval()
                    ref_loss = self.compute_reference_loss()
                    act_loss = self.compute_activation_loss()
                    self.model.train()
                    if linear_probing or freeze_bn or set_bn_eval:
                        # Set batch normalization to eval mode
                        self.model.apply(utils.set_bn_eval)
                self.log_loss.append({
                    'epoch': epoch,
                    'batch': i // self.sbatch,
                    'ref_loss': ref_loss.item(),
                    'projection': projected,
                    'act_loss': act_loss.item()
                })

            if log_accs and (epoch == 0 or (i // self.sbatch) % 30 == 0):
                # Log accuracies on previous and current tasks
                self.model.eval()
                total_batches = len(r) // self.sbatch + (1 if len(r) % self.sbatch > 0 else 0)
                print(f'Epoch {epoch}, batch {i // self.sbatch}/{total_batches}, evaluation...')
                accs = []
                for u in range(t+1):
                    xtest = data[u]['test']['x'].cuda()
                    ytest = data[u]['test']['y'].cuda()
                    _, valid_acc = self.eval(u, xtest, ytest)
                    accs.append(valid_acc)
                self.log_accs.append({
                    'epoch': epoch,
                    'batch': i // self.sbatch,
                    **{f'task_{u}': accs[u] for u in range(t + 1)}
                })
                self.model.train()
                if linear_probing or freeze_bn or set_bn_eval:
                    # Set batch normalization to eval mode
                    self.model.apply(utils.set_bn_eval)

        if defend:
            # Log the number of projections
            total_batches = len(r) // self.sbatch + (1 if len(r) % self.sbatch > 0 else 0)
            self.log_proj.append({
                'epoch': epoch,
                'num_projections': num_projections,
                'total_batches': total_batches
            })
        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r):
                b=r[i:i+self.sbatch]
            else:
                if self.drop_last_batch:
                    continue
                b=r[i:]
            images=x[b].to(device)  
            targets=y[b].to(device) 

            # Forward
            
            output = self.model.forward(images)[t]
            
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self, t, output, targets, apply_reg=True):
        # Regularization for all previous tasks
        loss_reg=0
        if apply_reg and t > 0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

            # print('loss_reg is {}'.format(loss_reg))    
        return self.ce(output,targets)+self.lamb*loss_reg