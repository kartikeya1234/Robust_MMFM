# Code taken and adapted from https://github.com/wagnermoritz/GSE

import torch
from vlm_eval.attacks.attack import Attack
import math

class IHT(Attack):

    def __init__(self, model, targeted=False, img_range=(0, 1), steps=100, prox='hard',ver=False, lam=5e-5, mask_out='none',stepsize=0.015,eps=4./255.):
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.steps = steps
        self.stepsize = stepsize
        self.ver = ver
        self.lam = lam
        self.eps = eps
        if mask_out != 'none':
            self.mask_out = mask_out
        else:
            self.mask_out = None
        if prox == 'hard':
            self.Prox = self.hardprox
        else:
            raise NotImplementedError



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
    
    def __call__(self, img):
        
        for param in self.model.model.parameters():
            param.requires_grad = False

        img = img.to(self.device)
        mask_out = self._set_mask(img)       
        x = torch.zeros_like(img) # perturbation to optimize
        z = x.clone() # used for FISTA extrapolation
        t = 1
        if self.ver:
            print('')

        for i in range(self.steps):
                # compue gradient
            x.requires_grad = True
            loss = self.model(img + x).sum() if self.targeted else -self.model(img + x).sum()
            loss.backward()
            x_grad = x.grad.data * mask_out
            x = x.detach()

            if self.ver and i % 20 == 0:
                print(f'Iteration: {i+1}, Loss: {loss}\n', end='')

                # FISTA update
            with torch.no_grad():
                t_ = .5 * (1 + math.sqrt(1 + 4 * t ** 2))
                alpha = (t - 1) / t_
                t = t_
                z_ = self.Prox(x=x - self.stepsize * x_grad, 
                               lam=self.lam * self.stepsize, 
                               img=img, 
                               eps=self.eps
                )
                x = z_ + alpha * (z_ - z)
                x = torch.clamp(x,-self.eps,self.eps)
                z = z_.clone()
                x = torch.clamp(img + x, *self.img_range) - img

        if self.ver:
            print('')
        print(f"L0 pert norm: {x.norm(p=0)}")
        
        return (img + x * mask_out).detach(), x.norm(p=0).item()

    def hardprox(self, x, lam, img, eps):
        '''
        Computes the hard thresholding proximal operator of the the
        perturbation x.

        :x:   Perturbation after gradient descent step.
        :lam: Regularization parameter.
        '''
        x_proj = torch.clamp(x,-eps,eps)
        x_temp = torch.clamp(img + x_proj,*self.img_range)
        x_proj = x_temp - img
        return torch.where(x ** 2 - (x_proj - x) ** 2 > 2 * lam, x_proj, 0)
