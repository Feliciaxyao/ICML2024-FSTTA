# https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import torch
import gc
import torch.nn as nn
import torch.jit
import time
import torch.nn.functional as F



import random

from utils.misc_tta import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)

class FAST(nn.Module):
    """
    Our FAST UPDATE adapts a model based on entropy minimization and gradient accumulation during testing one sample.
    A model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, t, lr_fast, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "TTA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.t = t
        self.lr_fast = lr_fast

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = fast_forward_and_adapt(x, self.model, self.optimizer, self.t)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)



class SLOW(nn.Module):
    """
    Our SLOW UPDATE adapts a model based on parameter analysis and gradient accumulation during testing N samples.
    A model adapts itself by updating on every forward.
    """
    def __init__(self, model_s, model, optimizer_s, N, steps=1, episodic=False):
        super().__init__()
        self.model_s = model_s
        self.model = model
        self.optimizer = optimizer_s
        self.steps = steps
        assert steps > 0, "TTA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.N = N
        

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = slow_forward_and_adapt(x, self.model_s, self.model, self.optimizer, self.N)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)




@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def cos_sim(vec1, vec2):
    "calculate cosine similaritity distance between variances"
    return F.cosine_similarity(vec1, vec2, dim=0)


#Fast-----------------------------------------------------------------------

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def fast_forward_and_adapt(x, model, optimizer, t, lr_fast):
    M = 3 # Set the fast update step size
    var_h = 0 # Set the initial variance value of gradients.
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    torch.cuda.synchronize()
    # start = time.time()
    # params_stack = {key: {'value':[]} for key, _ in model.named_parameters()}
    outputs = model('navigation', x)
    out = outputs['fused_logits']
    #out = outputs['logits']

    out.register_hook(print_grad)

    # "Calculate the TTA entropy"
    # out_0 = torch.softmax(out, 1)
    # x_0 = torch.where(torch.isinf(out), torch.full_like(out, -1e10), out) # Replace negative infinity with -1e10
    # x3 = torch.log_softmax(x_0, 1)
    # loss = torch.sum(-out_0 * x3).mean(0) / M

    # Using torch.finfo and torch.clamp to handle potential infinity values
    min_val = torch.finfo(out.dtype).min
    max_val = torch.finfo(out.dtype).max
    out_clamped = torch.clamp(out, min=min_val, max=max_val)
    
    out_0 = torch.softmax(out_clamped, 1)
    x3 = torch.log_softmax(out_clamped, 1)
    loss = torch.sum(-out_0 * x3).mean(0) / M


    loss.requires_grad_()
    loss.backward(retain_graph=True)

    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print("nan gradient found")
            print("name:",name)
            print("param:",param.grad)
            raise SystemExit

    # "Gradient Accumulation"

    grad_stack = {name: [] for name, param in model.named_parameters() if 'layer_norm' in name}

    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print("nan gradient found")
            print("name:", name)
            print("param:", param.grad)
            raise SystemExit
        if param.grad is not None and 'layer_norm' in name:
            grad_stack[name].append(param.grad.view(-1))

    # grad_stack += loss.grad
    # m_loss += loss
    # params_stack = torch.stack(loss.grad)

    # Our Fast Update performs every M steps.
    # if (t + 1) % M == 0:
    #    r_grad = grad_stack / M # refernence direction of gradient
    #    loss_grad, var_n = fast_principal_grad(ref_grad=r_grad, params_stack=params_stack, mean_loss=m_loss)
    #    loss.grad = loss_grad.clone()
        # Dynamic learning rate Scaling
    #     var_h, lr_fast_n = dynamic_lr_scaling(var_h, var_n, lr_fast)
        # Set the new learning rate for fast update
    #     for params in optimizer.param_groups:                       
    #         params['lr'] = lr_fast_n            
    #     optimizer.step()
    #     optimizer.zero_grad()       
        #end1 = time.time()
    #     params_stack = 0
    #     m_loss = 0

    if (t + 1) % M == 0:
        for name, grads in grad_stack.items():
            if len(grads) == M:
                r_grad = torch.stack(grads) / M
                loss_grad, var_n = fast_principal_grad(ref_grad=r_grad, params_stack=grads, ref_loss=loss)
                param = dict(model.named_parameters())[name]
                param.grad.copy_(loss_grad.view(param.size()))

        var_h, lr_fast_n = dynamic_lr_scaling(var_h, var_n, lr_fast)
        for params in optimizer.param_groups:                       
            params['lr'] = lr_fast_n
        optimizer.step()
        optimizer.zero_grad()
        grad_stack = {name: [] for name, param in model.named_parameters() if 'layer_norm' in name}

    return outputs, loss




def print_grad(grad):
    print(grad)



def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    

    for nm, m in model.named_modules(): 

        if 'vln_bert.embeddings' in nm:
            continue
        if 'vln_bert.lang_encoder' in nm:
            continue
        if 'vln_bert.img_embeddings' in nm:
            continue
        if 'vln_bert.local_encoder' in nm:
            continue
        if 'vln_bert.global_encoder' in nm:
            continue
        if 'vln_bert.sap_fuse_linear' in nm:
            continue
        if 'vln_bert.og_head' in nm:
            continue
        
        #if nm in ['norm']:
        #    continue
        #

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        #if isinstance(m , ('vln_bert.global_sap_head.net.2', 'vln_bert.local_sap_head.net.2')):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    # params_stack.update()
                    
    return params, names

    # for nm, m in model.named_modules():
        # if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            # for np, p in m.named_parameters():
                # if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    # params.append(p)
                    # names.append(f"{nm}.{np}")
    # return params, names


def stack_params(model):
    #params = []
    #names = []
    params_stack = {}

    for key, _ in model.named_parameters():
            if 'vln_bert.global_sap_head.net.2' in key:
                # params_stack.update([key]['value':[_]])
                params_stack[key]={'value':[_]}
            if 'vln_bert.local_sap_head.net.2' in key:
                params_stack[key]={'value':[_]}
                # params_stack.update([key]['value':[_]])

    return params_stack


def inner_params(model):
    #params = []
    #names = []
    params_stack = {}

    for key, _ in model.named_parameters():
            if 'vln_bert.global_sap_head.net.2' in key:
                # params_stack.update([key]['value':[_]])
                params_stack[key]=_
            if 'vln_bert.local_sap_head.net.2' in key:
                params_stack[key]=_
                # params_stack.update([key]['value':[_]])

    return params_stack


def save_state(model):  #..............................
    save_norm = {}
    for param_tensor in model.state_dict():
        if 'vln_bert.global_sap_head.net.2' in param_tensor:
            save_norm.update({param_tensor:model.state_dict()[param_tensor]})
        if 'vln_bert.local_sap_head.net.2' in param_tensor:
            save_norm.update({param_tensor:model.state_dict()[param_tensor]})
    PATH = './test_state_dict.pth'
    torch.save(save_norm, PATH)



def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state



def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with fstta."""
    # train mode, because tta optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what fstta updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tta."""
    is_training = model.training
    assert is_training, "tta needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tta needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tta should not update all params: " \
                               "check which require grad"
    # has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    has_ln = any([isinstance(m, nn.LayerNorm) for m in model.modules()])
    assert has_ln, "fstta needs normalization for its optimization"


# Operation for gradients--------------------------------------------------------------------------------------------

def transpose_pca(param_stack, comb=True):
    param_stack = torch.cat(param_stack)
    param_mean = param_stack.mean(dim=0)
    param_centered = param_stack - param_mean
    covariance = param_centered@param_centered.T/(param_stack.size(1)-1) # 
    pr_direction, pr_value, _ = torch.svd(covariance)
    pr_direction = param_centered.T@pr_direction
    pr_direction = pr_direction.norm(dim=0)/pr_direction
    return pr_direction, pr_value


def stack_and_pca(envs_weights, N):
    num_domains = N
    new_stack = [[] for _ in range(num_domains+1)]
    for i in range(num_domains+1):
        new_stack[i] = [envs_weights[ele]['value'][i].view(1, -1) for ele in envs_weights.keys()]
        new_stack[i] = torch.cat(new_stack[i], dim=1)
    pr_direction, pr_value = transpose_pca(new_stack)
    return pr_direction, pr_value

        
    


# r_grad, params_stack=params_stack, r_loss=r_loss
def fast_principal_grad(ref_grad, params_stack, ref_loss): 
    limit = 200  # Set truncation to prevent eigenvector coefficients from being too large
    if True:
        norm_ref = ref_grad.pow(2).sum()
        norm_ref = norm_ref.sqrt()
        # pr_vector, pr_value = transpose_pca(params_stack)
        g_basis, g_value = transpose_pca(params_stack)

        projections = []
        cali_grad = torch.zero_like(grad)

        # calculate projection
        for grad in params_stack:
            for basis_vector in g_basis:
                projection = torch.dot(grad, basis_vector)
                projections.append(projection)
                
                #calculate orthogonal decomposition
                # cali_grad = torch.zero_like(grad)
                for i in len(range(g_basis)):
                    if 1/g_value[i] > limit:
                        g_value[i] = 1/limit
                    # length calibration-1
                    g_proj = g_basis[i] * projections[i] / g_value[i]
                    # direction calibration
                    grad_mask = (ref_grad.flatten().unsqueeze(1) * g_proj).sum(0)
                    cali_mask = 2*(grad_mask > 0).float()-1
                    cali_proj = cali_mask * g_proj
                    cali_grad += cali_proj
                
        # length normalization???
        pr_grad = cali_grad / cali_grad.norm(dim=0)
        # pr_grad = pr_grad / pr_grad.norm(dim=0)
        # calculate 
        var = g_value.sum(0)
        # length calibration-2
        pr_grad = norm_ref * pr_grad
        # pr_grad = norm_ref * cali_grad 
        # param_size = ref_loss.numel()
        value_grad = pr_grad.view(ref_loss.size())
        return value_grad, var
    

def dynamic_lr_scaling(var_history, var_now, lr_fast_0, rho=0.95):
    # Update learning rate in fast update phase according to the distance between 2 variances 
    lr_fast_now = lr_truncation(var_history, var_now) * lr_fast_0
    # Update historical variance with memory momentum
    var_history_ = rho * var_history + (1 - rho) * var_now
    return var_history_, lr_fast_now


def lr_truncation(x, y, tao=0.7):
    out = 1 + tao - abs(x-y)
    return out


#slow------------------------------------------------------------------------------------------------

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def slow_forward_and_adapt(x, model_s, model, optimizer, N):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    torch.cuda.synchronize()
    ref_diff = x
    params_stack = {key: {'value':[]} for key, _ in model_s.named_parameters()}
    optimizer.zero_grad()
    slow_principal_grad(model_s, model, params_stack, ref_diff, N)
    optimizer.step()






# def principal_grad(meta_weights, inner_weights, params_stack, network):
def slow_principal_grad(model_s, model, params_stack, ref_diff, N):    
    if True:
        # origin_weights = ParamDict(model_s)
        # update_weights = ParamDict(model)
        # diff_weights = origin_weights - update_weights
        diff_weights = ref_diff
        norm_diff = sum([ele.pow(2).sum() for ele in diff_weights.values()])
        norm_diff = norm_diff.sqrt()
        principle_dir, pr_value = stack_and_pca(params_stack, N)
        # length calibration
        principle_dir *= norm_diff 
        grad_mask = torch.zeros_like(pr_value)
        start_index = 0
        for name, value in model_s.named_parameters():
            param_size = value.numel()
            end_index = start_index+param_size
            pra_grad = principle_dir[start_index:end_index, :]
            cali_direction = diff_weights[name] 
            cali_mask = (cali_direction.flatten().unsqueeze(1)*pra_grad).sum(0)
            grad_mask += cali_mask
            start_index= end_index

        # direction calibration    
        cali_mask = 2*(grad_mask > 0).float()-1
        pra_grad = cali_mask*principle_dir 
        # normalization
        coef = pr_value/pr_value.norm()
        pra_grad =  (coef*pra_grad).sum(1)

        start_index = 0
        # Learning PrincipleGrad for model update
        for name, value in model_s.named_parameters():
            param_size = value.numel()
            end_index = start_index+param_size
            value_grad = pra_grad[start_index:end_index].view(value.size())
            start_index= end_index
            value.grad = value_grad.clone()
    


def predict(self, x):
    return self.network(x)