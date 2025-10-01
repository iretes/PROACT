import sys, os, time
import numpy as np
import pickle as pkl    
import utils
import torch
from approaches.arguments import get_args
from resnet import ResNet18


tstart = time.time()

def main(args):

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint != None:
        checkpoint_dict = pkl.load(open(args.checkpoint, 'rb'))

    if args.approach == 'afec_ewc' or args.approach == 'ancl_ewc' or args.approach == 'ewc' or args.approach == 'afec_rwalk' or args.approach == 'rwalk' or args.approach == 'afec_mas' or args.approach == 'mas' or args.approach == 'afec_si' or args.approach == 'si' or args.approach == 'ft' or args.approach == 'random_init' or args.approach == 'rwalk2':
        log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}_addnoise_{}'.format(args.date, args.experiment, args.approach,args.seed,
                                                                        args.lamb, args.lr, args.batch_size, args.nepochs, args.addnoise)
    elif args.approach == 'gs':
        log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_rho_{}_eta_{}_lr_{}_batch_{}_epoch_{}_addnoise_{}'.format(args.date, args.experiment,
                                                                                            args.approach, args.seed, 
                                                                                            args.lamb, args.mu, args.rho,
                                                                                            args.eta, args.lr, args.batch_size, args.nepochs, args.addnoise)
    elif args.approach == 'naive':
        log_name = '{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}_addnoise_{}'.format(args.date, args.experiment, args.approach, args.seed,
                                                                args.lr, args.batch_size, args.nepochs, args.addnoise)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print('[CUDA unavailable]'); sys.exit()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Args -- Experiment
    if args.experiment == 'split_cifar100':
        # from dataloaders import split_cifar100 as dataloader
        from approaches.data_utils import generate_split_cifar100_tasks
    elif args.experiment == 'split_mini_imagenet':
        from approaches.data_utils import generate_split_mini_imagenet_tasks
    elif args.experiment == 'split_tiny_imagenet':
        from approaches.data_utils import generate_split_tiny_imagenet_tasks


    # Args -- Approach

    if args.approach == 'afec_ewc':
        from approaches import afec_ewc as approach
    elif args.approach == 'ancl_ewc':
        from approaches import ancl_ewc as approach
    elif args.approach == 'ewc':
        from approaches import ewc as approach
    elif args.approach == 'rwalk':
        from approaches import rwalk as approach
    elif args.approach == 'mas':
        from approaches import mas as approach
    elif args.approach == 'naive':
        from approaches import naive as approach

    if args.linear_probing_epochs > args.nepochs:
        raise ValueError('Linear probing epochs cannot be greater than the number of epochs for training.')

    print('Load data...')

    if args.experiment == 'split_cifar100':
        order = np.arange(100)
        im_sz = 32
        emb_fact = 1    
        data, taskcla, inputsize, task_order = generate_split_cifar100_tasks(args.tasknum, args.seed, rnd_order=False, order=order)
    elif args.experiment == 'split_mini_imagenet':
        order = np.arange(100)  
            
        class_num = 100 // (args.tasknum)  
        im_sz = 84
        emb_fact = 1    

        order = np.arange(100)
        #home = os.path.expanduser('~')
        mini_root = os.path.join('./', 'data', 'miniImagenet' ) 
        data, taskcla, inputsize, task_order = generate_split_mini_imagenet_tasks(mini_root, task_num = args.tasknum, 
                                                                    rnd_order=False, order=order) 
        
    elif args.experiment == 'split_tiny_imagenet':
        order = np.arange(200)  
        #home = os.path.expanduser('~')  
        root_add = os.path.join('./', 'data', 'tiny-imagenet-200') 
        dataset_file = './data/tiny_imagenet.npz'
        data, taskcla, inputsize, task_order = generate_split_tiny_imagenet_tasks(task_num = args.tasknum, 
                                                                    rnd_order=False, save_data=False,
                                                                    dataset_file=dataset_file, 
                                                                    order=order, root_add=root_add)
        
        class_num = 200 // (args.tasknum)  
        im_sz = 64
        emb_fact = 9
        
    print('\nInput size =', inputsize, '\nTask info =', taskcla)


    ########################################################################################################################

    print('Inits...')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # NOTE: the line above could be replaced with the following lines
    # torch.set_default_dtype(torch.float32)
    # torch.set_default_device('cuda')

    nf = 32

    net = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()
    net_emp = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()


    ########################################################################################################################

    save_dict = {}  
    save_dict['scenario'] = args.scenario_name
    save_dict['model_type'] = 'resnet'    
    save_dict['dataset'] = args.experiment
    save_dict['class_num'] = data['ncla'] // args.tasknum  
    save_dict['bs'] = args.batch_size
    save_dict['lr'] = args.lr
    save_dict['n_epochs'] = args.nepochs
    save_dict['model'] = net.state_dict()
    save_dict['model_name'] = net.__class__.__name__
    save_dict['task_num'] = args.lasttask        
    save_dict['task_order'] = task_order
    save_dict['seed'] = args.seed    
    save_dict['emb_fact'] = emb_fact  
    save_dict['im_sz'] = inputsize[1]  

    cont_method_args = {'method': args.approach} 
    for tmp_key in args.__dict__.keys():    
        cont_method_args[tmp_key] = args.__dict__[tmp_key] 
    cont_method_args['clip_fisher'] = args.clipfisher

    save_dict['cont_method_args'] = cont_method_args    

    approach_args = {
        'model': net,
        'sbatch': args.batch_size,
        'lr': args.lr,
        'lr_factor': args.lr_factor,
        'nepochs': args.nepochs,
        'drop_last_batch': args.drop_last_batch,
        'linear_probing_epochs': args.linear_probing_epochs,
        'clipgrad': args.clip,
        'srefsample': args.inverted_sample_size,
        'srefbatch': args.inverted_batch_size,
        'lamb_act': args.lamb_act,
        'lamb_act_decay': args.lamb_act_decay,
        'lamb_act_decay_rate': args.lamb_act_decay_rate,
        'lamb_act_decay_step': args.lamb_act_decay_step,
        'args': args,
        'log_name': log_name,
    }
    # NOTE: regularization parameters may be different from those used to train the model on previous tasks
    if 'afec' in args.approach or 'ancl' in args.approach: 
        approach_args.update({
            'lamb': args.lamb,
            'lamb_emp': args.lamb_emp,
            'clipfisher': args.clipfisher,
            'empty_net': net_emp,
        })
    elif args.approach == 'ewc':
        approach_args.update({
            'lamb': args.lamb,
            'clipfisher': args.clipfisher,
        })
    elif args.approach == 'mas' or args.approach == 'rwalk':
        approach_args.update({
            'lamb': args.lamb,
        })

    appr = approach.Appr(**approach_args)

    if args.checkpoint is not None:
        appr.load_model(checkpoint_dict['pretrained_ckpt']['model'])   
        if 'afec' in args.approach or 'ancl' in args.approach:
            appr.load_emp_model(checkpoint_dict['pretrained_ckpt']['cont_method_args']['model_emp'])

        if args.init_acc:
            accs_tmp = []
            for u in range(checkpoint_dict['pretrained_ckpt']['task_num']):  
                xtest = data[u]['test']['x']
                ytest = data[u]['test']['y']
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                accs_tmp.append(test_acc *100)   
            
            with np.printoptions(precision=2, suppress=True):   
                print(np.array(accs_tmp) ) 
            
        

    print('-' * 100)
    relevance_set = {}

    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

    for t, ncla in taskcla:
        if args.checkpoint is not None and t < args.lasttask:
            print('Skip task {:2d} : {:15s}'.format(t, data[t]['name']))
            continue


        if t==1 and 'find_mu' in args.date:
            break

        if t == args.lasttask and args.checkpoint is None:  
            break
        
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        # Get data
        xtrain = data[t]['train']['x'].clone()
        xvalid = data[t]['test']['x'].clone()   

        ytrain = data[t]['train']['y'].clone()
        yvalid = data[t]['test']['y'].clone()

        if args.checkpoint is not None and args.addnoise == True:
            
            if args.uniform is True:
                print('Using uniform noise')
                if 'inj_data_idx' not in checkpoint_dict.keys():
                    all_noise = torch.rand_like(xtrain) * 2 * checkpoint_dict['delta'] - checkpoint_dict['delta']
                    if args.inj_rate < 1.0:
                        num_noisy = int(xtrain.shape[0] * args.inj_rate)
                        noisy_indices = torch.randperm(xtrain.shape[0], device=xtrain.device)[:num_noisy]
                        mask = torch.zeros_like(xtrain)
                        mask[noisy_indices] = 1
                        all_noise = all_noise * mask
                else:
                    print('Using uniform noise only on the injected data')
                    noise_prm = checkpoint_dict['rnd_idx_train']   
                    xtrain = xtrain[noise_prm]  
                    ytrain = ytrain[noise_prm]
                    all_noise = torch.zeros_like(xtrain)    
                    inj_idx = checkpoint_dict['inj_data_idx']   
                    print(f'number of noisy data : {len(inj_idx)}')
                    all_noise[inj_idx] = torch.rand_like(xtrain[inj_idx]) * 2 * checkpoint_dict['delta'] - checkpoint_dict['delta']

                xtrain = torch.clamp(xtrain + all_noise, 0, 1)  

                
            else:
                print('Using noise from checkpoint')
                all_noise = checkpoint_dict['latest_noise']   
                noise_prm = checkpoint_dict['rnd_idx_train']   
                xtrain = xtrain[noise_prm]
                ytrain = ytrain[noise_prm]
                if args.inj_rate < 1.0:
                    num_noisy = int(xtrain.shape[0] * args.inj_rate)
                    noisy_indices = torch.randperm(xtrain.shape[0], device=xtrain.device)[:num_noisy]
                    xtrain[noisy_indices] = torch.clamp(xtrain[noisy_indices] + all_noise[noisy_indices], 0, 1)
                else:
                    xtrain = torch.clamp(xtrain + all_noise, 0, 1)

        task = t
        defend = args.defend if t == args.lasttask else False # NOTE: in this setting, we only defend the last task

        # Train
        if args.approach == 'naive':
            appr.train(
                task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla, args.log_accs,
                defend, args.distill_folder, args.agem, args.act_reg, 
                args.log_loss, args.freeze_bn, args.set_bn_eval
            )
        else:
            appr.train(
                task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla, args.log_accs,
                defend, args.distill_folder, args.agem, args.reg_project, args.reg_turn_off_on_projection,
                args.act_reg, args.log_loss, args.freeze_bn, args.set_bn_eval
            )
        print('-' * 100)

        # Test
        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                        100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss
            
        # Save
        
        print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t,:t+1])))
        print('Save at ' + args.output)

        with np.printoptions(precision=2, suppress=True):   
            print(acc)

    if args.checkpoint is not None: 
        acc[:args.lasttask, :args.lasttask] = checkpoint_dict['pretrained_ckpt']['acc_mat'][:args.lasttask, :args.lasttask]
        
    # Done
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    if args.checkpoint is not None: 
        acc_mat_sace_name = args.checkpoint.split('/')[-1]
        #remove the .pkl extension
        acc_mat_sace_name = acc_mat_sace_name[:-4]

        if args.addnoise is False:
            method = 'clean'
        if args.addnoise and args.uniform:  
            method = 'uniform'
        elif args.addnoise and args.uniform is False:
            method = 'ours'    
        
        if args.output_dir is not None:
            np.save(os.path.join(args.output_dir, f'acc_mat_{method}.npy'), acc)
        else:
            np.save(f'acc_mat_{acc_mat_sace_name}_{method}.npy', acc)   



    bwt_before = np.mean((acc[args.lasttask-1] - np.diag(acc))[:args.lasttask-1][:-1])
    avg_acc_before  = np.mean(acc[args.lasttask-1, :args.lasttask])

    if args.checkpoint is not None: 
        bwt_after = np.mean((acc[-1] - np.diag(acc))[:-1])
        avg_acc_after  = np.mean(acc[-1][:-1])
        last_task_acc = acc[-1, -1] 
        print(f'After BWT : {bwt_after} After avg acc : {avg_acc_after} Last task acc : {last_task_acc}')   

        
    print(f'Before BWT : {bwt_before} Before avg acc : {avg_acc_before}')  

    save_dict['last_task'] = int(args.lasttask)
    save_dict['acc_mat'] = acc
    save_dict['avg_acc'] = np.mean(acc[-1, :args.lasttask])
    save_dict['bwt'] = bwt_before
    save_dict['model'] = net.state_dict()
    if 'afec' in args.approach or 'ancl' in args.approach:
        save_dict['cont_method_args']['model_emp'] = net_emp.state_dict()   

    # save_dict['optim'] = optim.state_dict()

    if args.checkpoint is None:
        if args.output_dir is not None:
            save_path = os.path.join(args.output_dir, 'checkpoint.pkl')
            pkl.dump(save_dict, open(save_path, 'wb'))
            config_file_path = os.path.join(args.output_dir, 'exp_config.yaml')
            utils.save_exp_config(save_dict, config_file_path)
        else:
            save_name = utils.generate_save_name(save_dict)
            #check if the file exists and add a number to the end if it does    
            if os.path.exists(f'{args.approach}_{save_name}.pkl'):
                print(f'File {args.approach}_{save_name}.pkl already exists. Saving with a different name.')
                if 'afec' not in args.approach and 'ancl' not in args.approach:
                    pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_fisherclip_{args.clipfisher}_{save_name}_1.pkl', 'wb'))
                else:
                    pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_lambemp_{args.lamb_emp}_fisherclip_{args.clipfisher}_{save_name}_1.pkl', 'wb'))

            else:
                if 'afec' not in args.approach and 'ancl' not in args.approach:
                    pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_fisherclip_{args.clipfisher}_{save_name}.pkl', 'wb'))
                else:
                    pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_lambemp_{args.lamb_emp}_fisherclip_{args.clipfisher}_{save_name}.pkl', 'wb'))


if __name__ == '__main__':
    args = get_args()
    main(args)
    