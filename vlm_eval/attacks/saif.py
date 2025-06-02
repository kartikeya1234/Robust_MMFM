# Code adapted from https://github.com/wagnermoritz/GSE

from vlm_eval.attacks.attack import Attack
import torch
import math
import time

class SAIF(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1), steps=200,
                 r0=1, ver=False, k=10000, eps=16./255., mask_out='none', **kwargs):
        '''
        Adapted from: https://github.com/wagnermoritz/GSE/tree/main
        Implementation of the sparse Frank-Wolfe attack SAIF
        https://arxiv.org/pdf/2212.07495.pdf

        args:
        model:         Callable, PyTorch classifier.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        steps:         Int, number of FW iterations.
        r0:            Int, parameter for step size computation.
        ver:           Bool, print progress if True.
        '''
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.steps = steps
        self.r0 = r0
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.ver = ver
        self.k = k
        self.eps = eps
        if mask_out != 'none':
            self.mask_out = mask_out
        else:
            self.mask_out = None

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

    def __call__(self, x):
        '''
        Perform the  attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        k:   Int, sparsity parameter,
        eps: Float, perturbation magnitude parameter.

        Returns a tensor of the same shape as x containing adversarial examples.
        '''
        assert x.shape[0] == 1, "Only support batch size 1 for now"



        for param in self.model.model.parameters():
            param.requires_grad = False

        B, C, H, W = x.shape[1], x.shape[3], x.shape[4], x.shape[5]
        x = x.to(self.device)
        batchidx = torch.arange(B).view(-1, 1)

        mask_out = self._set_mask(x)
        # compute p_0 and s_0
        x_ = x.clone()
        x_.requires_grad = True
        out = self.model(x_)
        loss = -out.sum() if not self.targeted else out.sum()
        x__grad = torch.autograd.grad(loss, [x_])[0].detach() * mask_out
        p = -self.eps * x__grad.sign()
        p = p.detach().half()
        ksmallest = torch.topk(-x__grad.view(B, -1), self.k, dim=1)[1]
        ksmask = torch.zeros((B, C * H * W), device=self.device)
        ksmask[batchidx, ksmallest] = 1
        s = torch.logical_and(ksmask.view(*x.shape), x__grad < 0).float()
        s = s.detach().half()

        r = self.r0


        for t in range(self.steps):
            if self.ver:
                print(f'\r Iteration {t+1}/{self.steps}', end='')
            p.requires_grad = True
            s.requires_grad = True

            D = self.Loss_fn(x, s, p, mask_out)
            D.backward()

            mp = p.grad * mask_out
            ms = s.grad * mask_out
            with torch.no_grad():
                # inf-norm LMO
                v = (-self.eps * mp.sign()).half()

                # 1-norm LMO
                ksmallest = torch.topk(-ms.view(B, -1), self.k, dim=1)[1]
                ksmask = torch.zeros((B, C * H * W), device=self.device)
                ksmask[batchidx, ksmallest] = 1
                ksmask = ksmask.view(*x.shape) * mask_out
                z = torch.logical_and(ksmask, ms < 0).float().half()
                # update stepsize until primal progress is made
                mu = 1 / (2 ** r * math.sqrt(t + 1))
                progress_condition = (self.Loss_fn(x, s + mu * (z - s), p + mu * (v - p), mask_out)
                    > D)

                while progress_condition:
                    r += 1
                    if r >= 50:
                        break
                    mu = 1 / (2 ** r * math.sqrt(t + 1))
                    progress_condition = (self.Loss_fn(x, s + mu * (z - s), p + mu * (v - p), mask_out)
                    > D)


                p = p + mu * (v - p)
                s = s + mu * (z - s)

                x_adv = torch.clamp(x + p, *self.img_range)
                p = x_adv - x

            if self.ver and t % 10 == 0:
                print(f" Loss: {D}")
        if self.ver:
            print('')
        return (x + s * p * mask_out).detach(), torch.norm(s*p,p=0).item()

    def Loss_fn(self, x, s, p, mask_out):
        out = self.model(x + s * p * mask_out).sum()
        if self.targeted:
            return out
        else:
            return -out
