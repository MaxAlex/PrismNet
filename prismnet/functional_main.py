import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler


from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np

import prismnet.model as prismnet_model_arch
from prismnet import train, validate, inference, log_print, compute_saliency, compute_saliency_img, compute_high_attention_region
#compute_high_attention_region

from prismnet.engine.train_loop import optimize_input

from prismnet.model.utils import GradualWarmupScheduler
from prismnet.loader import SeqicSHAPE
from prismnet.utils import datautils

from itertools import chain


def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")

def save_evals(out_dir, filename, dataname, predictions, label, met):
    evals_dir = datautils.make_directory(out_dir, "out/evals")
    metrics_path = os.path.join(evals_dir, filename+'.metrics')
    probs_path = os.path.join(evals_dir, filename+'.probs')
    with open(metrics_path,"w") as f:
        if "_reg" in filename:
            print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}".format(
                dataname,
                met.acc,
                met.auc,
                met.prc,
                met.tp,
                met.tn,
                met.fp,
                met.fn,
                met.avg[7],
                met.avg[8],
                met.avg[9],
            ), file=f)
        else:
            print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:d}\t{:d}".format(
                dataname,
                met.acc,
                met.auc,
                met.prc,
                met.tp,
                met.tn,
                met.fp,
                met.fn,
            ), file=f)
    with open(probs_path,"w") as f:
        for i in range(len(predictions)):
            print("{:.3f}\t{}".format(predictions[i, 0], label[i, 0]), file=f)
    print("Evaluation file:", metrics_path)
    print("Prediction file:", probs_path)

    return metrics_path, probs_path

def save_infers(out_dir, filename, predictions):
    evals_dir = datautils.make_directory(out_dir, "out/infer")
    probs_path = os.path.join(evals_dir, filename+'.probs')
    with open(probs_path, "w") as f:
        for i in range(len(predictions)):
            print("{:f}".format(predictions[i, 0]), file=f)
    print("Prediction file:", probs_path)

    return probs_path

def main():
    global writer, best_epoch
    # Training settings
    parser = argparse.ArgumentParser(description='Official version of PrismNet')
    # Data options
    parser.add_argument('--data_dir',       type=str, default="data", help='data path')
    parser.add_argument('--exp_name',       type=str, default="cnn", metavar='N', help='experiment name')
    parser.add_argument('--p_name',         type=str, default="TIA1_Hela", metavar='N', help='protein name')
    parser.add_argument('--out_dir',        type=str, default=".", help='output directory')
    parser.add_argument('--mode',           type=str, default="pu", help='data mode')
    parser.add_argument("--infer_file",     type=str, help="infer file", default="")
    # Training Hyper-parameter
    parser.add_argument('--arch',           default="PrismNet", help='network architecture')
    parser.add_argument('--lr_scheduler',   default="warmup", help=' lr scheduler: warmup/cosine')
    parser.add_argument('--lr',             type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size',     type=int, default=64, help='input batch size')
    parser.add_argument('--nepochs',        type=int, default=200, help='number of epochs to train')
    parser.add_argument('--pos_weight',     type=int, default=2, help='positive class weight')
    parser.add_argument('--weight_decay',   type=float, default=1e-6, help='weight decay, default=1e-6')
    parser.add_argument('--early_stopping', type=int, default=20, help='early stopping')
    # Training 
    parser.add_argument('--load_best',      action='store_true', help='load best model')
    parser.add_argument('--eval',           action='store_true', help='eval mode')
    parser.add_argument('--train',          action='store_true', help='train mode')
    parser.add_argument('--infer',          action='store_true', help='infer mode')
    parser.add_argument('--infer_test',     action='store_true', help='infer test from h5')
    parser.add_argument('--eval_test',      action='store_true', help='eval test from h5')
    parser.add_argument('--saliency',       action='store_true', help='compute saliency mode')
    parser.add_argument('--saliency_img',   action='store_true', help='compute saliency and plot image mode')
    parser.add_argument('--har',            action='store_true', help='compute highest attention region')
    # misc
    parser.add_argument('--tfboard',        action='store_true', help='tf board')
    parser.add_argument('--no-cuda',        action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--workers',        type=int, help='number of data loading workers', default=2)
    parser.add_argument('--log_interval',   type=int, default=100, help='log print interval')
    parser.add_argument('--seed',           type=int, default=1024, help='manual seed')
    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.mode == 'pu':
        args.nstr = 1
    else:
        args.nstr = 0

    # out dir
    data_path  = args.data_dir + "/" + args.p_name + ".h5"
    identity   = args.p_name+'_'+args.arch+"_"+args.mode
    datautils.make_directory(args.out_dir,"out/")
    model_dir  = datautils.make_directory(args.out_dir,"out/models")
    model_path = os.path.join(model_dir, identity+"_{}.pth")

    if args.tfboard:
        tfb_dir  = datautils.make_directory(args.out_dir,"out/tfb")
        writer = SummaryWriter(tfb_dir)
    else:
        writer = None
    # fix random seed
    fix_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    
    #train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path), \
    #    batch_size=args.batch_size, shuffle=True,  **kwargs)
    #test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True), \
    #    batch_size=args.batch_size*8, shuffle=False, **kwargs)
    #print("Train set:", len(train_loader.dataset))
    #print("Test  set:", len(test_loader.dataset))


    print("Network Arch:", args.arch)
    model = getattr(prismnet_model_arch, args.arch)(mode=args.mode)
    prismnet_model_arch.param_num(model)
    # print(model)

    if args.load_best:
        filename = model_path.format("best")
        print("Loading model: {}".format(filename))
        model.load_state_dict(torch.load(filename,map_location='cpu'))

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))


def init_torch_model(arch_name, mode, cuda_mode, model_file_path=None):
    print("Network Arch:", arch_name)
    model = getattr(prismnet_model_arch, arch_name)(mode=mode)
    prismnet_model_arch.param_num(model)

    if cuda_mode:
        assert(torch.cuda.is_available())
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if model_file_path is not None:
        filename = model_file_path.format("best")
        print("Loading model: {}".format(filename))
        model.load_state_dict(torch.load(filename))

    model = model.to(device)

    return model, device


def run_train(data_path, out_dir,
              starting_model=None,
              mode='pu',
              arch_name='PrismNet',
              # lr_scheduler='warmup', # Alternately 'cosine'?  ## Apparently unused?
              lr=0.0001,
              batch_size=64,
              nepochs=200,
              pos_weight=2,
              weight_decay=1e-6,
              early_stopping=20, ## Also apparently unused?
              tfboard=False,
              workers=6, cuda_mode=True,
              seed=0, use_structure=True
              ):
    fix_seed(seed)

    data_dir = os.path.dirname(data_path)
    p_name = os.path.basename(data_path)

    identity = p_name + "_" + arch_name + "_" + mode
    model_dir = datautils.make_directory(out_dir, "models")
    model_path = os.path.join(model_dir, identity + "_{}.pth")

    model, device = init_torch_model(arch_name, mode, cuda_mode, starting_model)

    train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path,
                                                          use_structure=use_structure),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=cuda_mode)

    test_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True,
                                                         use_structure=use_structure),
                                              batch_size=batch_size*8, shuffle=False,
                                              num_workers=workers, pin_memory=cuda_mode)
    print("Train set:", len(train_loader.dataset))
    print("Test  set:", len(test_loader.dataset))

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=8, total_epoch=float(nepochs), after_scheduler=None)

    best_auc = 0
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, nepochs + 1):
        t_met       = train(model, device, train_loader, criterion, optimizer, batch_size)
        v_met, _, _ = validate(model, device, test_loader, criterion)
        scheduler.step(epoch)
        lr = scheduler.get_lr()[0]
        color_best='green'
        if best_auc < v_met.auc:
            best_auc = v_met.auc
            best_acc = v_met.acc
            best_epoch = epoch
            color_best = 'red'
            filename = model_path.format("best")
            torch.save(model.state_dict(), filename)
        if epoch - best_epoch > early_stopping:
            print("Early stop at %d, %s "%(epoch, 'prism_exp'))
            break

        if tfboard and writer is not None:
            writer.add_scalar('loss/train', t_met.other[0], epoch)
            writer.add_scalar('acc/train', t_met.acc, epoch)
            writer.add_scalar('AUC/train', t_met.auc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('loss/test', v_met.other[0], epoch)
            writer.add_scalar('acc/test', v_met.acc, epoch)
            writer.add_scalar('AUC/test', v_met.auc, epoch)
        line='{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} lr: {:.6f}'.format(
            p_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
        log_print(line, color='green', attrs=['bold'])

        line='{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f})'.format(
            p_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc)
        log_print(line, color=color_best, attrs=['bold'])

    print("{} auc: {:.4f} acc: {:.4f}".format(p_name, best_auc, best_acc))

    return model_path


def run_seq_opt(data_path, data_output,
                starting_model,
                mode='pu',
                arch_name='PrismNet',
                # lr_scheduler='warmup', # Alternately 'cosine'?  ## Apparently unused?
                lr=0.0001,
                batch_size=64,
                nepochs=200,
                pos_weight=2,
                weight_decay=1e-6,
                early_stopping=20, ## Also apparently unused?
                tfboard=False,
                workers=6, cuda_mode=True,
                seed=0, use_structure=True,
                save_reference_input_to=None,
                ):
    fix_seed(seed)

    data_dir = os.path.dirname(data_path)
    p_name = os.path.basename(data_path)

    model, device = init_torch_model(arch_name, mode, cuda_mode, starting_model)

    train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path,
                                                          use_structure=use_structure),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=cuda_mode)

    test_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True,
                                                         use_structure=use_structure),
                                              batch_size=batch_size*8, shuffle=False,
                                              num_workers=workers, pin_memory=cuda_mode)
    print("Train set:", len(train_loader.dataset))
    print("Test  set:", len(test_loader.dataset))

    # return train_loader, test_loader

    # DataLoader objects don't store changes to the input (they re-load on each iter, I guess?)
    # Also we're not using train/test sets here (how would you even do that?) so all into one
    # list of Tensors
    # putative_targets = list(chain(*[x[1] for x in train_loader] + [x[1] for x in test_loader]))
    input_list = list(chain(*[x[0] for x in train_loader] + [x[0] for x in test_loader]))
    input_tensor = torch.stack(input_list, 0)
    print(input_tensor.shape)

    if save_reference_input_to:
        with open(save_reference_input_to, 'wb') as out:
            torch.save(input_tensor, out)

    # Set all targets to one, since we're optimizing for inputs that lead to one,
    # and/so we want good inputs to have low loss
    aspirational_targets = torch.ones(size = (input_tensor.shape[0], 1))

    input_tensor = input_tensor.to(device)
    aspirational_targets = aspirational_targets.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    optimizer = torch.optim.Adam([input_tensor], lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=8, total_epoch=float(nepochs), after_scheduler=None)

    # TODO: there probably actually is a notion of train+test that could be applied here; namely, test the
    # optimized inputs against a different model, to see how well the inputs generalize.

    best_auc = 0
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, nepochs + 1):
        t_met       = optimize_input(model, input_tensor, aspirational_targets, criterion, optimizer)
        scheduler.step(epoch)
        lr = scheduler.get_lr()[0]

        with open(data_output, 'wb') as out:
            torch.save(input_tensor, out)
            print("Wrote tensor to %s" % data_output)

        if tfboard and writer is not None:
            writer.add_scalar('loss/train', t_met.other[0], epoch)
            writer.add_scalar('acc/train', t_met.acc, epoch)
            writer.add_scalar('AUC/train', t_met.auc, epoch)
            writer.add_scalar('lr', lr, epoch)
        line='{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} lr: {:.6f}'.format(
            p_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
        log_print(line, color='green', attrs=['bold'])

    print("{} auc: {:.4f} acc: {:.4f}".format(p_name, best_auc, best_acc))

    return data_output

    
def run_eval(data_path, model_file, out_dir,
             mode='pu',
             arch_name='PrismNet',
             batch_size=64,
             pos_weight=2,
             workers=6, cuda_mode=True,
             seed=0, use_structure=True
             ):
    fix_seed(seed)
    p_name = os.path.basename(data_path)
    identity = p_name + "_" + arch_name + "_" + mode

    model, device = init_torch_model(arch_name, mode, cuda_mode, model_file)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    test_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True,
                                                         use_structure=use_structure),
                                              batch_size=batch_size * 8, shuffle=False,
                                              num_workers=workers, pin_memory=cuda_mode)

    print("Test  set:", len(test_loader.dataset))

    met, y_all, p_all = validate(model, device, test_loader, criterion)
    print("> eval {} auc: {:.4f} acc: {:.4f}".format(p_name, met.auc, met.acc))
    return save_evals(out_dir, identity, p_name, p_all, y_all, met)

def run_infer(infer_file, model_file,
              data_path, out_dir,
              mode='pu',
              arch_name='PrismNet',
              batch_size=64,
              pos_weight=2,
              workers=6, cuda_mode=True,
              seed=0, use_structure=True
              ):
    assert(os.path.exists(infer_file))
    fix_seed(seed)

    p_name = os.path.basename(data_path)

    identity = p_name + "_" + arch_name + "_" + mode

    model, device = init_torch_model(arch_name, mode, cuda_mode, model_file)

    test_loader = torch.utils.data.DataLoader(SeqicSHAPE(infer_file, is_infer=True,
                                                         use_structure=use_structure),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=cuda_mode)

    p_all = inference(model, device, test_loader)
    identity = identity+"_"+ os.path.basename(infer_file).replace(".txt","")
    return save_infers(out_dir, identity, p_all)

def run_saliency(infer_file, model_file,
                 data_path, out_dir,
                 mode='pu',
                 arch_name='PrismNet',
                 batch_size=64,
                 workers=6, cuda_mode=True,
                 seed=0, use_structure=True
                 ):
    assert(os.path.exists(infer_file))
    fix_seed(seed)

    p_name = os.path.basename(data_path)
    identity = p_name + "_" + arch_name + "_" + mode
    model, device = init_torch_model(arch_name, mode, cuda_mode, model_file)

    test_loader = torch.utils.data.DataLoader(SeqicSHAPE(infer_file, is_infer=True,
                                                         use_structure=use_structure),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=cuda_mode)
    return compute_saliency(model, device, test_loader, identity, batch_size, out_dir)


def run_saliency_img(infer_file, model_file,
                     label, out_dir,
                     mode='pu',
                     arch_name='PrismNet',
                     batch_size=64,
                     workers=6, cuda_mode=True,
                     seed=0, use_structure=True
                     ):
    assert(os.path.exists(infer_file))
    fix_seed(seed)

    identity = label + "_" + arch_name + "_" + mode
    model, device = init_torch_model(arch_name, mode, cuda_mode, model_file)

    test_loader = torch.utils.data.DataLoader(SeqicSHAPE(infer_file, is_infer=True,
                                                         use_structure=use_structure),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=cuda_mode)
    return compute_saliency_img(model, device, test_loader, identity, batch_size, out_dir)

def run_har(infer_file, model_file,
            data_path, out_dir,
            mode='pu',
            arch_name='PrismNet',
            batch_size=64,
            workers=6, cuda_mode=True,
            seed=0, use_structure=True
            ):
    assert(os.path.exists(infer_file))
    fix_seed(seed)

    p_name = os.path.basename(data_path)
    identity = p_name + "_" + arch_name + "_" + mode
    model, device = init_torch_model(arch_name, mode, cuda_mode, model_file)

    test_loader = torch.utils.data.DataLoader(SeqicSHAPE(infer_file, is_infer=True,
                                                         use_structure=use_structure),
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=cuda_mode)
    return compute_high_attention_region(model, device, test_loader, identity, batch_size, out_dir)


    
    

if __name__ == '__main__':
    main()
