import os
import argparse
from solver import Solver
from data_loader import get_loader, pre_RaFD
from torch.backends import cudnn
from misc import InceptionNet
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True
    config.log_dir=os.path.join(config.save_dir,'logs')
    config.model_save_dir=os.path.join(config.save_dir,'models')
    config.sample_dir=os.path.join(config.save_dir,'samples')
    config.result_dir=os.path.join(config.save_dir,'results')
    config.inc_net_dir=os.path.join(config.save_dir,'IncNet')
    
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.inc_net_dir):
        os.makedirs(config.inc_net_dir)
        
    # Data loader.
    celeba_loader = None
    rafd_loader = None
    mnist_loader = None
    if config.preprocess_rafd:
        pre_RaFD(config.rafd_image_dir)

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    
    if config.dataset == 'MNIST':
        transform_ = [transforms.Resize(config.image_size), transforms.ToTensor()]
        transform_ = transforms.Compose(transform_)
        dataset = dset.MNIST('./dataset', transform=transform_,download=True)
        mnist_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # Solver for training and testing StarGAN.
    solver = Solver(rafd_loader, mnist_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD','MNIST']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()
    elif config.mode=='calc_score':
        incNet=InceptionNet(config)
        if config.train_inc:
            incNet.train(config)
        else:
            solver.restore_model(config.resume_iters)
            score=incNet.score(solver.G)
            print("Score: ",score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=8, help='dimension of domain labels (1st dataset)')
    # parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    # parser.add_argument('--con_dim', type=int, default=2, help='number of continuous dimensions')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=650, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=1024, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_cls', type=float, default=2, help='weight for domain classification loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_MI', type=float, default=5, help='weight for MI loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both','MNIST'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test','calc_score'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--train_inc',type=str2bool,default=False)
    parser.add_argument('--pretrained_incNet',type=str,default=None)
    parser.add_argument('--preprocess_rafd',type=str2bool,default=False)
    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='../CelebA_nocrop/img_celeba')
    parser.add_argument('--attr_path', type=str, default='../celebA/Anno/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--save_dir', type=str, default='stargan')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    # print(config)
    main(config)