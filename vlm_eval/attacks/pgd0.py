# Code taken and adapted from https://github.com/wagnermoritz/GSE

from vlm_eval.attacks.attack import Attack
import torch
import numpy as np

class PGD0(Attack):
    def __init__(self, model, *args, img_range=(0, 1), k=5000, n_restarts=1,
                 targeted=False, iters=200, stepsize=120000/255.0, eps=4./255.,ver=False,mask_out='none',**kwargs):
        '''
        Implementation of the PGD0 attack https://arxiv.org/pdf/1909.05040
        Author's implementation: https://github.com/fra31/sparse-imperceivable-attacks/tree/master
        Addapted from: https://github.com/wagnermoritz/GSE/tree/main

        args:
        model:         Callable, PyTorch classifier.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        k:             Int, sparsity parameter.
        n_restarts:    Int, number of restarts from random perturbation.
        iters:         Int, number of gradient descent steps per restart.
        stepsize:      Float, step size for gradient descent.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.k = k
        self.n_restarts = n_restarts
        self.eps = eps
        self.iters = iters
        self.stepsize = stepsize
        if mask_out != 'none':
            self.mask_out = mask_out
        else:
            self.mask_out = None
        self.ver = ver

    def _set_mask(self, data):
        mask = torch.ones_like(data)
        if self.mask_out == 'context':
            mask[:, :-1, ...] = 0
        elif self.mask_out == 'query':
            mask[:, -1, ...] = 0
        elif isinstance(self.mask_out, int):
            mask[:, self.mask_out, ...] = 0
        elif self.mask_out is None:
            pass
        else:
            raise NotImplementedError(f'Unknown mask_out: {self.mask_out}')
        return mask


    def __call__(self, x, *args, **kwargs):
        '''
        Perform the PGD_0 attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''

        for param in self.model.model.parameters():
            param.requires_grad = False
        
        mask_out = self._set_mask(x)
        x = x.to(self.device)
        B, C, H, W = x.shape[1], x.shape[3], x.shape[4], x.shape[5]

        for _ in range(self.n_restarts):
            if not len(x):
                break
            eps = torch.full_like(x, self.eps)
            lb, ub = torch.maximum(-eps, -x),torch.minimum(eps, 1.0 - x) #self.img_range[0] - x, self.img_range[1] - x
            pert = (torch.clamp(x + (ub - lb) * torch.rand_like(x) + lb, *self.img_range) - x).view(B, C, H, W) * mask_out.view(B, C, H, W)
            pert = self.project_L0(pert, lb, ub) # pert is of the shape (B, C, H, W)

            for _ in range(self.iters):
                pert.requires_grad = True
                loss = self.lossfn(x=x, pert=pert.view(*x.shape), mask_out=mask_out)
                loss.backward()

                if self.ver and _ % 20 == 0:
                    print(f"Loss: {loss}, Iter: {_}")
                
                grad = pert.grad.data.view(B,C,H,W) * mask_out.view(B, C, H, W) # shape (B, C, H, W)
                with torch.no_grad():
                    grad /= grad.abs().sum(dim=(1,2,3), keepdim=True) + 1e-10
                    pert += (torch.rand_like(x) - .5).view(B, C, H, W) * 1e-12 - self.stepsize * grad
                    pert = self.project_L0(pert, lb, ub)
        
        return (x + pert.view(*x.shape) * mask_out).detach()
    

    def project_L0_sigma(self, pert, sigma, kappa, x_orig):

        B, C, H, W = pert.shape
        x = torch.clone(pert)
        p1 = (1.0 / torch.maximum(1e-12, sigma) * (x_orig > 0).float()) + \
             (1e12 * (x_orig == 0).float())
        p2 = (1.0 / torch.maximum(torch.tensor(1e-12), sigma)) * \
             (1.0 / torch.maximum(torch.tensor(1e-12), x_orig) - 1) * \
             (x_orig > 0).float() + 1e12 * (x_orig == 0).float() + 1e12 * (sigma == 0).float()
        lmbd_l = torch.maximum(-kappa, torch.amax(-p1, dim=1, keepdim=True))
        lmbd_u = torch.minimum(kappa, torch.amin(p2, dim=1, keepdim=True)) 

        lmbd_unconstr = torch.sum((pert - x_orig) * sigma * x_orig, dim=1, keepdim=True) / torch.clamp(torch.sum((sigma * x_orig) ** 2, dim=1, keepdim=True), min=1e-12)
        lmbd = torch.maximum(lmbd_l, torch.minimum(lmbd_unconstr, lmbd_u))
        return 0


    def project_L0(self, pert, lb, ub):
        '''
        Project a batch of perturbations such that at most self.k pixels
        are perturbed and componentwise there holds lb <= pert <= ub.
        '''
        
        B, C, H, W = pert.shape # Here, pert is of the shape B, C, H, W
        p1 = torch.sum(pert ** 2, dim=1)
        p2 = torch.clamp(torch.minimum(ub.view(B, C, H, W) - pert, pert - lb.view(B, C, H, W)), 0)
        p2 = torch.sum(p2 ** 2, dim=1)
        p3 = torch.topk(-1 * (p1 - p2).view(p1.size(0), -1), k=H*W-self.k, dim=-1)[1] 
        pert = torch.maximum(torch.minimum(pert, ub.view(B, C, H, W)), lb.view(B, C, H, W))
        pert[torch.arange(0, B).view(-1, 1), :, p3//W, p3%H] = 0  
        return pert
        
    def lossfn(self, x, pert, mask_out):
        '''
        Compute the loss at x.
        '''
        return (2 * self.targeted - 1) * self.model(x + pert * mask_out).sum()
