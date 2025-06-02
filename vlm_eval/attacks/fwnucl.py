# Code taken and adapted from https://github.com/wagnermoritz/GSE
import torch
import math
from vlm_eval.attacks.attack import Attack

class FWnucl(Attack):
    def __init__(self, model, *args, iters=200, img_range=(-1, 1), ver=False,
                 targeted=False, eps=5, mask_out='none',**kwargs):
        '''
        Implementation of the nuclear group norm attack.

        args:
        model:         Callable, PyTorch classifier.
        ver:           Bool, print progress if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        eps:           Float, radius of the nuclear group norm ball.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.iters = iters
        self.ver = ver
        self.eps = eps
        self.gr = (math.sqrt(5) + 1) / 2
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


    def __loss_fn(self, x):
        '''
        Compute loss depending on self.targeted.
        '''
        if self.targeted:
            return -self.model(x).sum()
        else:
            return self.model(x).sum()


    def __call__(self, x, *args, **kwargs):
        '''
        Perform the nuclear group norm attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''

        for param in self.model.model.parameters():
            param.requires_grad = False

        mask_out = self._set_mask(x)
        x = x.to(self.device)
        noise = torch.zeros_like(x)
        noise.requires_grad = True

        for t in range(self.iters):
            if self.ver:
                print(f'\rIteration {t+1}/{self.iters}', end='')
            
            loss = self.__loss_fn(x + noise * mask_out)
            loss.backward()
            noise.grad.data = noise.grad.data * mask_out
            s = self.__groupNuclearLMO(noise.grad.data, eps=self.eps)
            with torch.no_grad():
                gamma = self.__lineSearch(x=x, s=s, noise=noise)
                noise = (1 - gamma) * noise + gamma * s
            noise.requires_grad = True

            if self.ver and t % 20 == 0:
                print(f"Iteration: {t}, Loss: {loss.item()}")
        x = torch.clamp(x + noise, 0, 1)
        if self.ver:
            print("")
        return x.detach()


    def __lineSearch(self, x, s, noise, steps=25):
        '''
        Perform line search for the step size.
        '''
        a = torch.zeros(x.shape[1], device=self.device).view(-1, 1, 1, 1)
        b = torch.ones(x.shape[1], device=self.device).view(-1, 1, 1, 1)
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr
        sx = s - noise

        for i in range(steps):
            loss1 = self.__loss_fn(x + noise + (c * sx).view(*x.shape))
            loss2 = self.__loss_fn(x + noise + (d * sx).view(*x.shape))
            mask = loss1 > loss2

            b[mask] = d[mask]
            mask = torch.logical_not(mask)
            a[mask] = c[mask]

            c = b - (b - a) / self.gr
            d = a + (b - a) / self.gr

        return (b + a) / 2


    def __groupNuclearLMO(self, x, eps=5):
        '''
        LMO for the nuclear group norm ball.
        '''

        B, C, H, W = x.shape[1], x.shape[3], x.shape[4], x.shape[5]
        size = 32 if H > 64 else 4

        # turn batch of images into batch of size by size pixel groups per
        # color channel
        xrgb = [x.view(B, C, H, W)[:, c, :, :] for c in range(C)]
        xrgb = [xc.unfold(1, size, size).unfold(2, size, size) for xc in xrgb]
        xrgb = [xc.reshape(-1, size, size) for xc in xrgb]

        # compute nuclear norm of each patch (sum norms over color channels)
        norms = torch.linalg.svdvals(xrgb[0])
        for xc in xrgb[1:]:
            norms += torch.linalg.svdvals(xc)
        norms = norms.sum(-1).reshape(B, -1)

        # only keep the patch g* with the largest nuclear norm for each image
        idxs = norms.argmax(dim=1).view(-1, 1)
        xrgb = [xc.reshape(B, -1, size, size) for xc in xrgb]
        xrgb = [xc[torch.arange(B).view(-1, 1), idxs].view(B, size, size)
                for xc in xrgb]

        # build index tensor corr. to the position of the kept patches in x
        off = (idxs % (W / size)).long() * size
        off += torch.floor(idxs / (W / size)).long() * W * size
        idxs = torch.arange(0, size**2,
                            device=self.device).view(1, -1).repeat(B, 1) + off
        off = torch.arange(0, size,
                           device=self.device).view(-1, 1).repeat(1, size)
        off = off * W  - off * size
        idxs += off.view(1, -1)

        # compute singular vector pairs corresponding to largest singular value
        # and final perturbation (LMO solution)
        pert = torch.zeros_like(x).view(B, C, H, W)
        for i, xc in enumerate(xrgb):
            U, _, V = torch.linalg.svd(xc)
            U = U[:, :, 0].view(B, size, 1)
            V = V.transpose(-2, -1)[:, :, 0].view(B, size, 1)
            pert_gr = torch.bmm(U, V.transpose(-2, -1)).reshape(B, size * size)
            idx = torch.arange(B).view(-1, 1)
            pert_tmp = pert[:, i, :, :].view(B, -1)
            pert_tmp[idx, idxs] = pert_gr * eps
            pert_clone = pert.clone()
            pert_clone[:, i, :, :] = pert_tmp.view(B, H, W)

        return pert_clone.view(*x.shape)
