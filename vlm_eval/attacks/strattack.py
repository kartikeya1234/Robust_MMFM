# Code taken and adapted from https://github.com/wagnermoritz/GSE

from vlm_eval.attacks.attack import Attack
import torch
import math
import torch.nn.functional as F

class StrAttack(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(0, 1), kappa=0,
                 max_iter=100, ver=False, search_steps=2, max_c=1e10, rho=1, mask_out='none',
                 c=2.5, retrain=False, **kwargs):
        '''
        Implementation of StrAttack: https://arxiv.org/abs/1808.01664
        Adapted from https://github.com/KaidiXu/StrAttack

        args:
        model:         Callable, PyTorch classifier.
        targeted:      Bool, given label is used as a target label if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        max_iter:      Int, number of iterations.
        ver:           Bool, print progress if True.
        search_steps:  Int, number of binary search steps.
        max_c:         Float, upper bound for regularizaion parameter.
        rho:           Float, ADMM parameter.
        c:             Float, initial regularization parameter.
        '''
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.max_iter = max_iter
        self.ver = ver
        self.search_steps = search_steps
        self.max_c = max_c
        self.rho = rho
        self.c = c
        self.retrain = retrain
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

    def __call__(self, imgs, *args, **kwargs):
        '''
        Perform StrAttack on a batch of images x with corresponding labels y.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.

        Returns a tensor of the same shape as x containing adversarial examples
        '''

        for param in self.model.model.parameters():
            param.requires_grad = False

        c_ = self.c
        imgs = imgs.to(self.device)
        sh = imgs.shape
        batch_size = sh[1]
        mask_out = self._set_mask(imgs)

        alpha, tau, gamma = 5, 2, 1
        eps = torch.full_like(imgs, 1.0) * mask_out
        # 16 for imagenet, 2 for CIFAR and MNIST
        filterSize = 8 if sh[-1] > 32 else 2
        stride = filterSize
        # convolution kernel used to compute norm of each group
        slidingM = torch.ones((1, sh[3], filterSize, filterSize), device=self.device)

        cs = torch.ones(batch_size, device=self.device) * c_
        lower_bound = torch.zeros(batch_size)
        upper_bound = torch.ones(batch_size) * self.max_c

        o_bestl2 = torch.full_like(torch.randn(batch_size), 1e10, dtype=torch.float)
        o_bestscore = torch.full_like(o_bestl2, -1, dtype=torch.float)
        o_bestattack = imgs.clone()
        o_besty = torch.ones_like(imgs)

        for step in range(self.search_steps):

            bestl2 = torch.full_like(o_bestl2, 1e10, dtype=torch.float)
            bestscore = torch.full_like(o_bestl2, -1, dtype=torch.float)

            z, v, u, s = (torch.zeros_like(imgs) for _ in range(4))

            for iter_ in range(self.max_iter):
                if (not iter_%10 or iter_ == self.max_iter - 1) and self.ver:
                    print(f'\rIteration: {iter_+1}/{self.max_iter}, ' +
                          f'Search Step: {step+1}/{self.search_steps}', end='')

                # first update step (7) / Proposition 1
                delta = self.rho / (self.rho + 2 * gamma) * (z - u / self.rho)

                b = (z - s / self.rho) * mask_out
                tmp = torch.minimum(self.img_range[1] - imgs, eps)
                w = torch.where(b.view(*sh) > tmp.view(*sh), tmp, b) # creating issue (1x5x'5'x3x224x224 instead of 1x5x1x3x224x224)
                tmp = torch.maximum(self.img_range[0] - imgs, -eps)
                w = torch.where(b.view(*sh) < tmp.view(*sh), tmp, w)
                
                c = z - v / self.rho
                cNorm = torch.sqrt(F.conv2d(c.view(sh[1], sh[3], sh[4], sh[5]) ** 2, slidingM, stride=stride))
                cNorm = torch.where(cNorm == 0, torch.full_like(cNorm, 1e-12), cNorm)
                cNorm = F.interpolate(cNorm, scale_factor=filterSize)
                y = torch.clamp((1 - tau / (self.rho * cNorm.unsqueeze(0).unsqueeze(3))), 0) * c
                
                # second update step (8) / equation (15)
                z_grads = self.__get_z_grad(imgs, z.clone(), cs)
                eta = alpha * math.sqrt(iter_ + 1)
                coeff = (1 / (eta + 3 * self.rho))
                z = coeff * (eta * z + self.rho * (delta + w + y) + u + s + v - z_grads)

                # third update step (9)
                u = u + self.rho * (delta - z) * mask_out
                v = v + self.rho * (y - z) * mask_out
                s = s + self.rho * (w - z) * mask_out
                # get info for binary search
                x = imgs + y * mask_out
                l2s = torch.sum((z ** 2).reshape(z.size(1), -1), dim=-1)
                
                for i, (l2, x_) in enumerate(zip(l2s, x.squeeze(0))):
                    if l2 < bestl2[i]:
                        bestl2[i] = l2
                    if l2 < o_bestl2[i]:
                        o_bestl2[i] = l2
                        o_bestattack[:,i] = x_.detach().unsqueeze(0).clone()
                        o_besty[:,i] = y[:,i]
            for i in range(batch_size):
                
                lower_bound[i] = max(lower_bound[i], cs[i])
                if upper_bound[i] < 1e9:
                    cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    cs[i] *= 5

        del v, u, s, z_grads, w, tmp
        
        if self.retrain:
            cs = torch.full_like(o_bestl2, 5.0, dtype=torch.float)
            zeros = torch.zeros_like(imgs)

            for step in range(8):
                bestl2 = torch.full_like(cs, 1e10, dtype=torch.float, device=self.device)
                bestscore = torch.full_like(cs, -1, dtype=torch.float, device=self.device)

                Nz = o_besty[o_besty != 0]
                e0 = torch.quantile(Nz.abs(), 0.03)
                A2 = torch.where(o_besty.abs() <= e0, 0, 1)
                z1 = o_besty
                u1 = torch.zeros_like(imgs)
                tmpc = self.rho / (self.rho + gamma / 100)

                for j in range(100):
                    if self.ver and not j % 10:
                        print(f'\rRetrain iteration: {step+1}/8, ' +
                              f'Search Step: {j+1}/200', end='')

                    tmpA = (z1 - u1) * tmpc
                    tmpA1 = torch.where(o_besty.abs() <= e0, zeros, tmpA)
                    cond = torch.logical_and(tmpA >
                                             torch.minimum(self.img_range[1] - imgs, eps),
                                             o_besty.abs() > e0)
                    tmpA2 = torch.where(cond, torch.minimum(self.img_range[1] - imgs, eps),
                                        tmpA1)
                    cond = torch.logical_and(tmpA <
                                             torch.maximum(self.img_range[0] - imgs, -eps),
                                             o_besty.abs() > e0)
                    deltA = torch.where(cond, torch.maximum(self.img_range[0] - imgs, -eps),
                                        tmpA2)
                    
                    x = imgs + deltA * mask_out
                    grad = self.__get_z_grad(imgs, deltA, cs)

                    stepsize = 1 / (alpha + 2 * self.rho)
                    z1 = stepsize * (alpha * z1 * self.rho
                                     * (deltA + u1) - grad * A2)
                    u1 = u1 + deltA - z1

                    for i, (l2, x_) in enumerate(zip(l2s, x.squeeze(0))):
                        if l2 < bestl2[i]:
                            bestl2[i] = l2
                            #bestscore[i] = asc
                        if l2 < o_bestl2[i]:
                            o_bestl2[i] = l2
                            #o_bestscore[i] = asc
                            o_bestattack[:,i] = x_.detach().unsqueeze(0).clone()
                            o_besty[i] = deltA[i]


                for i in range(batch_size):
                    if (bestscore[i] != -1 and bestl2[i] == o_bestl2[i]):
                        upper_bound[i] = min(upper_bound[i], cs[i])
                        if upper_bound[i] < 1e9:
                            cs[i] = (lower_bound[i] + upper_bound[i]) / 2

                    else:
                        lower_bound[i] = max(lower_bound[i], cs[i])
                        if upper_bound[i] < 1e9:
                            cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                        else:
                            cs[i] *= 5

        if self.ver:
            print('')
        
        return (o_bestattack * mask_out).detach()


    def __get_z_grad(self, imgs, z, cs):
        '''
        Compute and return gradient of loss wrt. z.
        '''
        z.requires_grad = True
        tmp = self.model(z + imgs).sum() if self.targeted else -self.model(z + imgs).sum()
        loss = torch.mean(cs.to(self.device) * tmp)
        z_grad_data = torch.autograd.grad(loss, [z])[0].detach()
        z.detach_()
        return z_grad_data
