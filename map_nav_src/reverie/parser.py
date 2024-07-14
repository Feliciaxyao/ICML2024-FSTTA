import argparse
import os
import math


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='reverie', choices=['reverie'])
    parser.add_argument('--output_dir', type=str, default='../datasets/REVERIE/exprs_map/finetune/dagger-vitbase-seed.12_6', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--fusion', default='dynamic', choices=['global', 'local', 'avg', 'dynamic'])
    parser.add_argument('--dagger_sample', default='sample', choices=['sample', 'expl_sample', 'argmax'])
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--loss_nav_3', action='store_true', default=False)

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=100000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--eval_first', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=200) # 200
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--max_objects', type=int, default=20) # 20
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default='../datasets/REVERIE/exprs_map/finetune/dagger-vitbase-seed.3/ckpts/best_val_unseen_3', help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--multi_endpoints", default=True, action="store_true")
    parser.add_argument("--multi_startpoints", default=False, action="store_true")
    parser.add_argument("--aug_only", default=False, action="store_true")
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default='../datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker/ckpts/model_step_100000.pt', help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01) 

    parser.add_argument("--features", type=str, default='vitbase')
    parser.add_argument('--obj_features', type=str, default='vitbase')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)

    parser.add_argument('--enc_full_graph', default=True, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=True)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5) #origin=0.5 cannnot be set as 1, it will hurt the model's capability.
    parser.add_argument('--feat_dropout', type=float, default=0.4) #origin=0.4: if defalut=1, the results decades seriously. 


    # Submision configuration
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument("--submit", action='store_true', default=True)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=True)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")# origin=0.00001 
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=768)
    parser.add_argument('--obj_feat_size', type=int, default=768)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument("--gamma", default=0, type=float, help='reward discount factor') #0.9
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='dagger'
    )

    # new for fstta-------------------------------------------------------------------------------------------
    #parser = argparse.ArgumentParser(description='SAR exps')

    
    # Exp Settings
    parser.add_argument('--method', default='sar', type=str, help='no_adapt, tent, eata, sar')
    parser.add_argument('--model', default='vitbase_timm', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='bs1', type=str, help='normal, mix_shifts, bs1, label_shifts')

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(1000)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order). See Section 4.3 for details;')
    parser.add_argument('--TTA_lr', default=0.064/64, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    #--------------------------------------------------------------------------------------------------------------------------------------------------new



    args, _ = parser.parse_known_args()

    args = postprocess_args(args)


    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])

    obj_ft_file_map = {
        'vitbase': 'obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5',
    }
    args.obj_ft_file = os.path.join(ROOTDIR, 'REVERIE', 'features', obj_ft_file_map[args.obj_features])
    
    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'REVERIE', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

