import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='pmnist', type=str, required=False,
                        choices=[ 'split_cifar10_100',
                                  'split_cifar100',
                                  'split_cifar100_SC',
                                  'split_mini_imagenet', 
                                  'split_tiny_imagenet'],
                        help='(default=%(default)s)')
    
    parser.add_argument('--approach', default='lrp', type=str, required=False,
                        choices=['afec_ewc', 'ancl_ewc', 'ewc', 'rwalk', 'mas', 'naive'], help='(default=%(default)s)')
    
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=20, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=16, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--rho', default=0.3, type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float, help='(default=%(default)f)')
    parser.add_argument('--eta', default=0.8, type=float, help='(default=%(default)f)')
    parser.add_argument('--smax', default=400, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb_emp', default='0', type=float, help='(default=%(default)f)')
    parser.add_argument('--nu', default='0.1', type=float, help='(default=%(default)f)')
    parser.add_argument('--mu', default=0, type=float, help='groupsparse parameter')

    parser.add_argument('--img', default=0, type=float, help='image id to visualize')

    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--lasttask', type=int, help='(default=%(default)s)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to support coefficient')

    parser.add_argument('--scenario_name', type = str)
    parser.add_argument('--checkpoint', default=None , type = str)
    parser.add_argument('--addnoise', action='store_true')
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--l2normal', action='store_true')
    parser.add_argument('--blend', action='store_true')
    parser.add_argument('--rndnewds', action='store_true')
    parser.add_argument('--newds', action='store_true')
    parser.add_argument('--rndtopknoise', action='store_true')
    parser.add_argument('--init_acc', action='store_true')
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--pattern_add', type = str)

    parser.add_argument('--clip', default=100., type=float)
    parser.add_argument('--optim', type=str, default='sgd')

    parser.add_argument('--distill_folder', type=str, default=None, help='Folder to save distillation data')
    parser.add_argument('--defend', action='store_true', help='Use defense mechanism')
    parser.add_argument('--agem', action='store_true', help='Use AGEM (Averaged Gradient Episodic Memory)')
    parser.add_argument('--inverted_batch_size', default=128, type=int, required=False, help='Batch size for inverted data (default=%(default)d)')
    parser.add_argument('--inverted_sample_size', default=256, type=int, required=False, help='Sample size of inverted data (default=%(default)d)')
    parser.add_argument('--reg_project', action='store_true', help='Use regularization projection')
    parser.add_argument('--reg_turn_off_on_projection', action='store_true', help='Turn off regularization on projection')
    parser.add_argument('--act_reg', action='store_true', help='Use activation regularization')
    parser.add_argument('--lamb_act', default=0.1, type=float, required=False, help='Lambda for activation regularization (default=%(default)f)')
    parser.add_argument('--lamb_act_decay', action='store_true', help='Use decay for activation regularization')
    parser.add_argument('--lamb_act_decay_rate', default=0.1, type=float, required=False, help='Decay rate for activation regularization (default=%(default)f)')
    parser.add_argument('--lamb_act_decay_step', default=5, type=int, required=False, help='Decay step for activation regularization (default=%(default)d)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output files')
    parser.add_argument('--log_accs', action='store_true', help='Log accuracies during training')
    parser.add_argument('--drop_last_batch', action='store_true', help='Drop the last incomplete batch')
    parser.add_argument('--clipfisher', action='store_true', help='Clip Fisher information')
    parser.add_argument('--linear_probing_epochs', default=0, type=int, required=False, help='Number of initial epochs for linear probing (must be < nepochs)')
    parser.add_argument('--lr_factor', default=1.0, type=float, required=False, help='Factor to reduce learning rate after linear probing epochs (default=1, no reduction)')
    parser.add_argument('--log_loss', action='store_true', help='Log reference loss during training')
    parser.add_argument('--freeze_bn', action='store_true', help='Freeze BatchNorm layers during training')
    parser.add_argument('--set_bn_eval', action='store_true', help='Set BatchNorm layers to eval mode during training')

    parser.add_argument('--inj_rate', type=float, default=1.0, help='Injection rate for noise (default=1.0)')

    if sys.argv[0] == 'activations.py' or sys.argv[0] == 'activations_inverted.py' or sys.argv[0] == 'gradients.py':
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args

