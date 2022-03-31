from __future__ import print_function
import argparse, os, copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import prismnet.model as arch
from prismnet.utils import log_print, metrics, datautils
    
def train(model, device, train_loader, criterion, optimizer, batch_size):
    model.train()
    met = metrics.MLMetrics(objective='binary')
    for batch_idx, (x0, y0) in enumerate(train_loader):
        x, y = x0.float().to(device), y0.to(device).float()
        if y0.sum() ==0 or y0.sum() == batch_size:
            continue
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        prob = torch.sigmoid(output)

        y_np = y.to(device='cpu', dtype=torch.long).detach().numpy()
        p_np = prob.to(device='cpu').detach().numpy()
        met.update(y_np, p_np,[loss.item()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return met


def optimize_input(model, input_tensor, target_tensor, criterion, optimizer):
    model.train(False)  # Not setting to train mode, setting to eval mode (??????)
    met = metrics.MLMetrics(objective='binary')

    print(input_tensor.shape)
    input_tensor.requires_grad_(True)
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    prob = torch.sigmoid(output)

    y_np = target_tensor.to(device='cpu', dtype=torch.long).detach().numpy()
    p_np = prob.to(device='cpu').detach().numpy()
    met.update(y_np, p_np, [loss.item()])
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(input_tensor.parameters(), 5)
    optimizer.step()

    return met

def validate(model, device, test_loader, criterion):
    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            #if y0.sum() ==0:
            #    import pdb; pdb.set_trace()
            output  = model(x)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)

            y_np = y.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)
    
    met = metrics.MLMetrics(objective='binary')
    met.update(y_all, p_all,[l_all.mean()])
    

    
    return met, y_all, p_all

def inference(model, device, test_loader):
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            output = model(x)
            prob = torch.sigmoid(output)

            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

    p_all = np.concatenate(p_all)
    return p_all


def compute_saliency(model, device, test_loader, identity, batch_size, out_dir, only_seq):
    from prismnet.model import GuidedBackpropSmoothGrad

    model.eval()

    saliency_dir = datautils.make_directory(out_dir, "out/saliency")
    saliency_path = os.path.join(saliency_dir, identity+'.sal')

    # sgrad = SmoothGrad(model, device=device)
    sgrad = GuidedBackpropSmoothGrad(model, device=device, only_seq=only_seq)
    sal = ""
    for batch_idx, (x0, y0) in enumerate(test_loader):
        X, Y = x0.float().to(device), y0.to(device).float()
        output = model(X)
        prob = torch.sigmoid(output)
        p_np = prob.to(device='cpu').detach().numpy().squeeze()

        if len(p_np.shape) == 0:
            p_np = p_np.reshape(-1)

        guided_saliency = sgrad.get_batch_gradients(X, Y)
        # import pdb; pdb.set_trace()
        N, NS, _, _ = guided_saliency.shape # (N, 101, 1, 5)
        
        for i in range(N):
            inr = batch_idx*batch_size + i
            str_sal = datautils.mat2str(np.squeeze(guided_saliency[i]))
            sal += "{}\t{:.6f}\t{}\n".format(inr, p_np[i], str_sal)
            
    f = open(saliency_path,"w")
    f.write(sal)
    f.close()
    print(saliency_path)

    return saliency_path


def compute_saliency_img(model, device, test_loader, identity, batch_size, out_dir):
    from prismnet.model import GuidedBackpropSmoothGrad
    from prismnet.utils import visualize

    def saliency_img(X, mul_saliency, outdir="results"):
        """generate saliency image

        Args:
            X ([np.ndarray]): raw input(L x 5/4)
            mul_saliency ([np.ndarray]): [description]
            outdir (str, optional): [description]. Defaults to "results".
        """
        if X.shape[-1]==5:
            x_str = X[:,4:]
            str_null = np.zeros_like(x_str)
            ind =np.where(x_str == -1)[0]
            str_null[ind,0]=1
            
            ss = mul_saliency[:,:]
            s_str = mul_saliency[:,4:]
            s_str = (s_str - s_str.min())/(s_str.max() - s_str.min())
            ss[:,4:] = s_str * (1-str_null)

            str_null=np.squeeze(str_null).T
        else:
            str_null = None
            ss = mul_saliency[:,:]

        visualize.plot_saliency(
            X.T, 
            ss.T, 
            nt_width=100, 
            norm_factor=3, 
            str_null=str_null, 
            outdir=outdir
        )


    prefix_n = len(str(len(test_loader.dataset)))
    datautils.make_directory(out_dir, "out/imgs/")
    imgs_dir = datautils.make_directory(out_dir, "out/imgs")
    # imgs_path = imgs_dir+'/{:0'+str(prefix_n)+'d}_{:.3f}..pdf'

    saliency_path = os.path.join(imgs_dir, 'all.sal')

    # sgrad = SmoothGrad(model, device=device)
    sgrad = GuidedBackpropSmoothGrad(model, device=device, magnitude=1)
    for batch_idx, (x0, y0) in enumerate(test_loader):
        X, Y = x0.float().to(device), y0.to(device).float()
        output = model(X)
        prob = torch.sigmoid(output)
        p_np = prob.to(device='cpu').detach().numpy().squeeze()

        if len(p_np.shape) == 0:  # Using squeeze() was NOT THE CORRECT WAY
            p_np = p_np.reshape(-1)

        guided_saliency  = sgrad.get_batch_gradients(X, Y)
        mul_saliency = copy.deepcopy(guided_saliency)
        mul_saliency[:,:,:,:4] =  guided_saliency[:,:,:,:4] * X[:,:,:,:4]
        N, NS, _, _ = guided_saliency.shape # (N, 101, 1, 5)
        sal = ""
        for i in tqdm(range(N)):
            inr = batch_idx*batch_size + i
            str_sal = datautils.mat2str(np.squeeze(guided_saliency[i]))
            sal += "{}\t{:.6f}\t{}\n".format(inr, p_np[i], str_sal)
            # img_path = imgs_path.format(inr, p_np[i])
            # import pdb; pdb.set_trace()
            img_path = os.path.join(imgs_dir, '%s__%d.pdf' % (identity, i))
            saliency_img(
                X[i,0].to(device='cpu').detach().numpy(), 
                mul_saliency[i,0].to(device='cpu').numpy(), 
                outdir=img_path)


    f = open(saliency_path,"w")
    f.write(sal)
    f.close()
    print(saliency_path)

    return saliency_path



def compute_high_attention_region(model, device, test_loader, identity, batch_size, out_dir):
    from prismnet.model import GuidedBackpropSmoothGrad

    har_dir = datautils.make_directory(out_dir, "out/har")
    har_path = os.path.join(har_dir, identity+'.har')

    L = 20
    har = ""
    # sgrad = SmoothGrad(model, device=device)
    sgrad = GuidedBackpropSmoothGrad(model, device=device)
    for batch_idx, (x0, y0) in enumerate(test_loader):
        X, Y = x0.float().to(device), y0.to(device).float()
        output = model(X)
        prob = torch.sigmoid(output)
        p_np = prob.to(device='cpu').detach().numpy().squeeze()

        if len(p_np.shape) == 0:
            p_np = p_np.reshape(-1)

        guided_saliency  = sgrad.get_batch_gradients(X, Y)
        
        attention_region = guided_saliency.sum(dim=3)[:,0,:].to(device='cpu').numpy() # (N, 101, 1)
        N,NS = attention_region.shape # (N, 101)
        for i in range(N):
            inr = batch_idx*batch_size + i
            iar = attention_region[i]
            ar_score = np.array([ iar[j:j+L].sum() for j in range(NS-L+1)])
            # import pdb; pdb.set_trace()
            highest_ind = np.argmax(iar)
            har += "{}\t{:.6f}\t{}\t{}\n".format(inr, p_np[i], highest_ind, highest_ind+L)

    f = open(har_path, "w")
    f.write(har)
    f.close()
    print(har_path)

    return har_path

