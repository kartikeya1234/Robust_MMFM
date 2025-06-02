# Code taken and adapted from https://github.com/wagnermoritz/GSE
import torch
import torchvision
import math
import torch.nn.functional as F

from vlm_eval.attacks.attack import Attack


# required input size : batch_size x num_media x num_frames x channels x height x width 
class GSEAttack(Attack):
    def __init__(self, model, *args, mask_out='none',ver=False, img_range=(-1, 1), search_steps=4,
                 targeted=False, sequential=False, search_factor=2,
                 gb_size=5, sgm=1.5, mu=1, sigma=0.0025, iters=200, k_hat=10,
                 q=0.25, **kwargs):
        '''
        Implementation of the GSE attack.

        args:
        model:         Callable, PyTorch classifier.
        mask_out:      Masks out context images if set to context, query images if set to query and none if set to none.
        ver:           Bool, print progress if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        search_steps:  Int, number of steps for line search on the trade-off
                       parameter.
        targeted:      Bool, given label is used as a target label if True.
        sequential:    Bool, perturbations are computed sequentially for all
                       images in the batch if True. For fair comparison to
                       Homotopy attack.
        search_factor: Float, factor to increase/decrease the trade-off
                       parameter until an upper/lower bound for the line search
                       is found.
        gb_size:       Odd int, size of the Gaussian blur kernel.
        sgm:           Float, sigma of the gaussian blur kernel
        mu:            Float, trade-off parameter for 2-norm regularization.
        sigma:         Float, step size
        iters:         Int, number of iterations.
        k_hat:         Int, number of iterations before transitioning to NAG.
        q:             Float, inverse of increase factor for adjust_lambda.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.ver = ver
        self.search_steps = search_steps
        self.sequential = sequential
        self.search_factor = search_factor
        self.gb_size = gb_size
        self.sgm = sgm
        self.mu = mu
        self.sigma = sigma
        self.iters = iters
        self.k_hat = k_hat
        self.q = q
        if mask_out != 'none':
            self.mask_out = mask_out
        else:
            self.mask_out = None

    def adjust_lambda(self, lam, noise):
        '''
        Adjust trade-off parameters (lambda) to update search space.
        '''
        x = noise.detach().clone().abs().mean(dim=1, keepdim=True).sign()
        gb = torchvision.transforms.GaussianBlur((self.gb_size, self.gb_size),
                                                 sigma=self.sgm)
        x = gb(x) + 1
        x = torch.where(x == 1, self.q, x)
        lam /= x[:, 0, :, :]
        return lam


    def section_search(self, x, steps=50):
        '''
        Section search for finding the maximal lambda such that the
        perturbation is non-zero after the first iteration.
        '''

        noise = torch.zeros_like(x, requires_grad=True) # the shape of 'x' is batch_size x num_media x num_frames x Color x height x width
        loss = (-self.model(x + noise).sum() + self.mu
                * torch.norm(noise.view(x.size(1), x.size(3), x.size(4), x.size(5)), p=2, dim=(1,2,3)).sum())
        grad = torch.autograd.grad(loss, [noise])[0].detach()
        noise.detach_()
        ones = torch.ones_like(x.view(x.size(1), x.size(3), x.size(4), x.size(5)))[:, 0, :, :]

        # define upper and lower bound for line search
        lb = torch.zeros((x.size(1),), dtype=torch.float,
                         device=self.device).view(-1, 1, 1)
        ub = lb.clone() + 0.001
        mask = torch.norm(self.prox(grad.clone().view(x.size(1),x.size(3),x.size(4),x.size(5)) * self.sigma,
                                      ones * ub * self.sigma),
                          p=0, dim=(1,2,3)) != 0
        while mask.any():
            ub[mask] *= 2
            mask = torch.norm(self.prox(grad.clone().view(x.size(1),x.size(3),x.size(4),x.size(5)) * self.sigma,
                                          ones * ub * self.sigma),
                              p=0, dim=(1,2,3)) != 0

        # perform search
        for _ in range(steps):
            cur = (ub + lb) / 2
            mask = torch.norm(self.prox(grad.clone().view(x.size(1),x.size(3),x.size(4),x.size(5)) * self.sigma,
                                          ones * cur * self.sigma),
                              p=0, dim=(1,2,3)) == 0
            ub[mask] = cur[mask]
            mask = torch.logical_not(mask)
            lb[mask] = cur[mask]
        cur = (lb + ub).view(-1) / 2
        return 0.01 * cur


    def __call__(self, x, y, *args, **kwargs):
        '''
        Call the attack for a batch of images x or sequentially for all images
        in x depending on self.sequential.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        if self.sequential:
            result = x.clone()
            for i, (x_, y_) in enumerate(zip(x, y)):
                result[i] = self.perform_att(x_.unsqueeze(0),
                                             y_.unsqueeze(0),
                                             mu=self.mu, sigma=self.sigma,
                                             k_hat=self.k_hat).detach()
            return result
        else:
            return self.perform_att(x, y, mu=self.mu, sigma=self.sigma,
                                    k_hat=self.k_hat)


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


    def perform_att(self, x, mu, sigma, k_hat):
        '''
        Perform GSE attack on a batch of images x with corresponding labels y.
        '''
        x = x.to(self.device)
        B, C, H, W = x.shape[1], x.shape[3], x.shape[4], x.shape[5] # Input is of the shape Batch x Num_media x num_frames x colors x height x width
        lams = self.section_search(x)
        mask_out = self._set_mask(x).view(B,C,H,W)
        # save x, y, and lams for resetting them at the beginning of every
        # section search step
        save_x = x.clone()
        save_lams = lams.clone()
        # upper and lower bounds for section learch
        ub_lams = torch.full_like(lams, torch.inf)
        lb_lams = torch.full_like(lams, 0.0)
        # tensor for saving succesful adversarial examples in inner loop
        result = x.clone()
        # tensor for saving best adversarial example so far
        result2 = x.clone()
        best_l0 = torch.full((B,), torch.inf, device=self.device).type(x.type())

        # section search
        for step in range(self.search_steps):
            x = save_x.clone()
            lams = save_lams.clone()
            lam = torch.ones_like(x.view(B, C, H, W))[:, 0, :, :] * lams.view(-1, 1, 1)
            # tensor for tracking for which images adv. examples have been found
            active = torch.ones(B, dtype=bool, device=self.device)
            # set initial perturbation to zero
            noise = torch.zeros_like(x, requires_grad = True)
            noise_old = noise.clone()
            lr = 1

            # attack
            for j in range(self.iters):
                if self.ver:
                    print(f'\rSearch step {step + 1}/{self.search_steps}, ' +
                          f'Prox.Grad. Iteration {j + 1}/{self.iters}, ' +
                          f'Images left: {x.shape[1]}', end='')
                if len(x) == 0:
                    break

                self.model.model.zero_grad()
                loss = (-self.model(x + noise).sum() + mu
                        * (torch.norm(noise.view(B, C, H, W), p=2, dim=(1,2,3)) ** 2).sum())
                noise_grad_data = torch.autograd.grad(loss, [noise])[0].detach().view(B, C, H, W)
                #print(f"{loss} {(torch.norm(noise.view(B, C, H, W), p=2, dim=(1,2,3)) ** 2).sum()}")
                with torch.no_grad():
                    
                    noise_grad_data = noise_grad_data * mask_out # Mask_out shape B x C x H x W
                    lr_ = (1 + math.sqrt(1 + 4 * lr**2)) / 2
                    if j == k_hat:
                        lammask = (lam > lams.view(-1, 1, 1))[:, None, :, :]
                        lammask = lammask.repeat(1, C, 1, 1)
                        noise_old = noise.clone()
                    if j < k_hat:
                        noise = noise - sigma * noise_grad_data.view(1, B, 1, C, H, W)
                        noise = self.prox(noise.view(B, C, H, W), lam * sigma).view(1, B, 1, C, H, W)
                        noise_tmp = noise.clone()
                        noise = lr / lr_ * noise + (1 - (lr/ lr_)) * noise_old
                        noise_old = noise_tmp.clone()
                        lam = self.adjust_lambda(lam, noise.view(B, C, H, W))
                    else:
                        noise = noise - sigma * noise_grad_data.view(1, B, 1, C, H, W)
                        noise_tmp = noise.clone()
                        noise = lr / lr_ * noise + (1 - (lr/ lr_)) * noise_old
                        noise_old = noise_tmp.clone()
                        noise[lammask.view(1, B, 1, C, H, W)] = 0
                    # clamp adv. example to valid range
                    x_adv = torch.clamp(x + noise, *self.img_range)
                    noise = x_adv - x
                    lr = lr_
                    

                noise.requires_grad = True
            
            # section search
            # no adv. example found => decrease upper bound and current lambda
            # adv. example found => save it if the "0-norm" is better than of the
            # previous adv. example, increase lower bound and current lambda
            for i in range(B):
                if active[i]:
                    ub_lams[i] = save_lams[i]
                    save_lams[i] = 0.95 * lb_lams[i] + 0.05 * save_lams[i]
                else:
                    print("here")
                    l0 = self.l20((result[i] - save_x[i]).unsqueeze(0)).to(self.device)
                    if l0 < best_l0[i]:
                        best_l0[i] = l0
                        result2[i] = result[i].clone()
                    if torch.isinf(ub_lams[i]):
                        lb_lams[i] = save_lams[i]
                        save_lams[i] *= self.search_factor
                    else:
                        lb_lams[i] = save_lams[i]
                        save_lams[i] = (ub_lams[i] + save_lams[i]) / 2

        if self.ver:
            print('')
        
        return x_adv

    def extract_patches(self, x):
        '''
        Extracts and returns all overlapping size by size patches from
        the image batch x.
        '''
        B, C, _, _ = x.shape
        size = 8
        kernel = torch.zeros((size ** 2, size ** 2))
        kernel[range(size**2), range(size**2)] = 1.0
        kernel = kernel.view(size**2, 1, size, size)
        kernel = kernel.repeat(C, 1, 1, 1).to(x.device)
        out = F.conv2d(x, kernel, groups=C)
        out = out.view(B, C, size, size, -1)
        out = out.permute(0, 4, 1, 2, 3)
        return out.contiguous()
    
    def l20(self, x):
        '''
        Computes d_{2,0}(x[i]) for all perturbations x[i] in the batch x
        as described in section 3.2.
        '''
        B, N, M, C, _, _ = x.shape
        l20s = []
    
        for b in range(B):
            for n in range(N):
                for m in range(M):
                    x_ = x[b, n, m]  # Select the specific perturbation x[b, n, m]
                    patches = self.extract_patches(x_.unsqueeze(0))  # Add unsqueeze to match 6D input
                    l2s = torch.norm(patches, p=2, dim=(2,3,4))
                    l20s.append((l2s != 0).float().sum().item())
    
        return torch.tensor(l20s)


    def prox(self, grad_loss_noise, lam):
        '''
        Computes the proximal operator of the 1/2-norm of the gradient of the
        adversarial loss wrt current noise.
        '''

        lam = lam[:, None, :, :]
        sh = list(grad_loss_noise.shape)
        lam = lam.expand(*sh)

        p_lam = (54 ** (1 / 3) / 4) * lam ** (2 / 3)

        mask1 = (grad_loss_noise > p_lam)
        mask2 = (torch.abs(grad_loss_noise) <= p_lam)
        mask3 = (grad_loss_noise < -p_lam)
        mask4 = mask1 + mask3

        phi_lam_x = torch.arccos((lam / 8) * (torch.abs(grad_loss_noise) / 3)
                                 ** (-1.5))

        grad_loss_noise[mask4] = ((2 / 3) * torch.abs(grad_loss_noise[mask4])
                                  * (1 + torch.cos((2 * math.pi) / 3
                                  - (2 * phi_lam_x[mask4]) / 3))).to(torch.float32)
        grad_loss_noise[mask3] = -grad_loss_noise[mask3]
        grad_loss_noise[mask2] = 0

        return grad_loss_noise
