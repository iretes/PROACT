import sys, time, os
import numpy as np
import torch
from copy import deepcopy
import approaches.utils as utils    
sys.path.append('..')
from approaches.arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
import yaml
import pandas as pd
args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Appr(object):

    def __init__(
        self, model, lamb, nepochs=100, sbatch=256, lr=0.001, lr_min=2e-6,
        lr_factor=1, lr_patience=5, clipgrad=100, drop_last_batch=False,
        linear_probing_epochs=0,
        args=None, log_name = None,
        srefsample=256, srefbatch=128,
        lamb_act=1, lamb_act_decay=False, lamb_act_decay_rate=0.1, lamb_act_decay_step=5
        ):
        self.model=model
        self.model_old=model

        self.nepochs = nepochs
        self.linear_probing_epochs = linear_probing_epochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min *1/3
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
        print(f'lamb is: {lamb}')
        self.alpha = 0.9
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        self.log_name = log_name
        
        self.s = {}
        self.s_running = {}
        self.fisher = {}
        self.fisher_running = {}
        self.p_old = {}
        
        self.eps = 0.01
        
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.s[n] = 0
                self.s_running[n] = 0
                self.fisher[n] = 0
                self.fisher_running[n] = 0
                self.p_old[n] = p.data.clone()
        
        
        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        return optimizer
        
    def register_fisher_into_buffer(self):
        for n,_ in self.model.named_parameters():
            if 'heads' not in n:
                self.model.register_buffer('fisher_{}'.format(n.replace('.', '_')), self.fisher[n])
                self.model.register_buffer('running_fisher_{}'.format(n.replace('.', '_')), self.fisher_running[n]) 
    
    def rigister_s_into_buffer(self):
        for n,_ in self.model.named_parameters():
            if 'heads' not in n:
                self.model.register_buffer('s_{}'.format(n.replace('.', '_')), self.s[n])
                self.model.register_buffer('running_s_{}'.format(n.replace('.', '_')), self.s_running[n])

    def register_dummies_into_buffer(self):
        for n, p in self.model.named_parameters():
            if 'heads' not in n:
                self.model.register_buffer('fisher_{}'.format(n.replace('.', '_')), torch.zeros_like(p))
                self.model.register_buffer('running_fisher_{}'.format(n.replace('.', '_')), torch.zeros_like(p))
                self.model.register_buffer('s_{}'.format(n.replace('.', '_')), torch.zeros_like(p))
                self.model.register_buffer('running_s_{}'.format(n.replace('.', '_')), torch.zeros_like(p))

            

    def load_from_buffers(self):
        for n, p in self.model.named_parameters():
            if 'heads' not in n:
                self.fisher[n] = getattr(self.model, 'fisher_{}'.format(n.replace('.', '_')))   
                self.fisher_running[n] = getattr(self.model, 'running_fisher_{}'.format(n.replace('.', '_')))
                self.s[n] = getattr(self.model, 's_{}'.format(n.replace('.', '_')))
                self.s_running[n] = getattr(self.model, 'running_s_{}'.format(n.replace('.', '_')))


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
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        if freeze_bn:
            # Freeze BatchNorm layers
            self.model.apply(utils.freeze_bn_params)
        lr = self.lr
        patience = self.lr_patience
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
            clock0=time.time()
            num_batch = xtrain.size(0)

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
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print()
            #save log for current task & old tasks at every epoch
            
            
            # if valid_loss < best_loss:
            #     best_loss = valid_loss
            #     best_model = utils.get_model(self.model)
            #     patience = self.lr_patience
            #     print(' *', end='')

            best_model = utils.get_model(self.model)

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
        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        utils.freeze_model(self.model_old) # Freeze the weights

        if self.nepochs > self.linear_probing_epochs:
            # Update fisher & s
            for n,p in self.model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        self.fisher[n] = self.fisher_running[n].clone()
                        self.s[n] = (1/2) * self.s_running[n].clone()
                        self.s_running[n] = self.s[n].clone()

            self.register_fisher_into_buffer()  
            self.rigister_s_into_buffer()   

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
            if dotg < 0:
                projected = True
                grad_cur = grad_cur - (dotg / torch.dot(grads_ref, grads_ref)) * grads_ref
        
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
                with torch.no_grad():
                # Add regularization term gradient
                    for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                        if param.grad is not None:
                            reg_grad = 2 * self.lamb * (self.fisher[name] + self.s[name]) * (param - param_old)
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
            images=x[b]
            targets=y[b]

            images, targets = images.cuda(), targets.cuda()
            
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
            
            if not linear_probing:
                # Compute Fisher & s
                self.update_fisher_and_s()

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
            images=x[b]
            targets=y[b]

            images, targets = images.cuda(), targets.cuda() 
            
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
        if apply_reg and t>0:
            
            for (n,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum((self.fisher[n] + self.s[n])*(param_old-param).pow(2))
        return self.ce(output,targets)+self.lamb*loss_reg
    
    def update_fisher_and_s(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    # Compute running fisher
                    fisher_current = p.grad.data.pow(2)
                    self.fisher_running[n] = self.alpha*fisher_current + (1-self.alpha)*self.fisher_running[n]

                    # Compute running s
                    loss_diff = -p.grad * (p.detach() - self.p_old[n])
                    fisher_distance = (1/2) * (self.fisher_running[n]*(p.detach() - self.p_old[n])**2)
                    s = loss_diff /(fisher_distance+self.eps)
                    self.s_running[n] = self.s_running[n] + s

                self.p_old[n] = p.detach().clone()