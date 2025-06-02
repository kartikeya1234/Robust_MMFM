# Code taken and adapted from https://github.com/wagnermoritz/GSE
from vlm_eval.attacks.attack import Attack
import torch

class SparseRS(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1),
                 n_queries=10000, k=100, n_restarts=10, alpha_init=0.8, mask_out='none',**kwargs):
        '''
        Implementation of the L0 variant SparseRS https://arxiv.org/abs/2006.12834
        Authors' implementation: https://github.com/fra31/sparse-rs
        Adapted from: https://github.com/wagnermoritz/GSE/tree/main
        
        args:
        model:         Callable, PyTorch classifier.
        targeted:      Bool, given label is used as a target label if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        n_queries:     Int, max number of queries to the model
        k:             Int, initial sparsity parameter
        n_restarts:    Int, number of restarts with random initialization
        alpha_init:    Float, inital value for alpha schedule
        '''
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.n_queries = n_queries
        self.k = k
        self.n_restarts = n_restarts
        self.alpha_init = alpha_init
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


    def __call__(self, x, *args, **kwargs):
        '''
        Perform SparseRS L0 on a batch of images x with corresponding labels y.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''

        for param in self.model.model.parameters():
            param.requires_grad = False

        torch.random.manual_seed(0)
        torch.cuda.random.manual_seed(0)
        x = x.to(self.device)

        with torch.no_grad():
            for _ in range(self.n_restarts):
                if len(x) == 0:
                    break

                x_adv = self.__perturb(x.clone())

        return x_adv.detach()
    

    def __perturb(self, x):
        '''
        Perform the attack from a random starting point.
        '''
        mask_out = self._set_mask(x)
        B, C, H, W = x.shape[1], x.shape[3], x.shape[4], x.shape[5]
        batchidx = torch.arange(B, device=self.device).view(-1, 1)
        result = x.clone().view(B, C, H, W)

        # M: set of perturbed pixel indices, U_M: set of unperturbed pixel indices
        batch_randperm = torch.rand(B, H * W, device=self.device).argsort(dim=1)
        M = batch_randperm[:, :self.k]
        U_M = batch_randperm[:, self.k:]
        result[batchidx, :, M//W, M%H] = self.__sampleDelta(B, C, self.k)

        best_loss = self.__lossfn(result.view(*x.shape))

        for i in range(1, self.n_queries):
            if B == 0:
                break
            # reset k_i currently perturbed pixels and perturb k_i new pixels
            k_i = max(int(self.__alphaSchedule(i) * self.k), 1)
            A_idx = torch.randperm(self.k, device=self.device)[:k_i]
            B_idx = torch.randperm(H * W - self.k, device=self.device)[:k_i]
            A_set, B_set = M[:, A_idx], U_M[:, B_idx]

            z = result.clone()
            z[batchidx, :, A_set//W, A_set%H] = x.view(B, C, H, W)[batchidx, :, A_set//W, A_set%H]
            if k_i > 1:
                z[batchidx, :, B_set//W, B_set%H] = self.__sampleDelta(B, C, k_i)
            else: # if only one pixel is changed, make sure it actually changes
                new_color = self.__sampleDelta(B, C, k_i)
                while (mask := (z[batchidx, :, B_set//W, B_set%H] == new_color).view(B, -1).all(dim=-1)).any():
                    new_color[mask] = self.__sampleDelta(mask.int().sum().item(), C, k_i)
                z[batchidx, :, B_set//W, B_set%H] = new_color

            # save perturbations that improved the loss/margin
            loss = self.__lossfn(z, y)
            mask = loss < best_loss
            best_loss[mask] = loss[mask]
            mask = torch.logical_or(mask, margin < -1e-6)
            if mask.any():
                #best_margin[mask] = margin[mask]
                tmp = result[active]
                tmp[mask] = z[mask]
                result[active] = tmp
                U_M[mask.nonzero().view(-1, 1), B_idx] = A_set[mask]
                M[mask.nonzero().view(-1, 1), A_idx] = B_set[mask]
            
            # stop working on successful adv examples
            mask = best_margin < 0
            if mask.any():
                mask = torch.logical_not(mask)
                active[active.clone()] = mask
                x, y, z, M, U_M = x[mask], y[mask], z[mask], M[mask], U_M[mask]
                best_margin, best_loss = best_margin[mask], best_loss[mask]
                B = len(y)
                batchidx = torch.arange(B, device=self.device).view(-1, 1)

        return result


    def __sampleDelta(self, B, C, k):
        '''
        Sample k-pixel perturbations for B images. Each pixel is assigned a
        random corner in the C-dimensional cube defined by self.img_range.
        '''
        fac = self.img_range[1] - self.img_range[0]
        return self.img_range[0] + fac * torch.randint(0, 1, [B, k, C],
                                                       dtype=torch.float,
                                                       device=self.device)
    

    def __alphaSchedule(self, iteration):
        '''
        Update number of pixels to perturb based in the current iteration.
        '''
        iteration = int(iteration / self.n_queries * 10000)
        factors = [1, 2, 4, 5, 6, 8, 10, 12, 15, 20]
        alpha_schedule = [10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000]
        idx = bisect.bisect_left(alpha_schedule, iteration)
        return self.alpha_init / factors[idx]
    

    def __lossfn(self, x):
        '''
        Compute the loss depending on self.targeted.
        '''
        return self.model(x).sum() if self.targeted else -self.model(x).sum()
