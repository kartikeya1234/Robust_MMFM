# Code taken and adapted from https://github.com/wagnermoritz/GSE
import torch
from vlm_eval.attacks.attack import Attack

class EAD(Attack):

    def __init__(self,model, targeted=False, img_range=(0,1), steps=100, beta=5e-5, mask_out='none', ver=False, binary_steps=2, step_size=1e-2, decision_rule='L1'):
        
        super().__init__(model=model, targeted=targeted, img_range=img_range)
        self.steps = steps
        self.ver = ver
        self.binary_steps = binary_steps
        self.beta = beta
        if mask_out != 'none':
            self.mask_out = mask_out
        else:
            self.mask_out = None
        self.decision_rule = decision_rule
        self.ver = ver
        self.step_size = step_size

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

    def __call__(self, x_orig):
        
        for param in self.model.model.parameters():
            param.requires_grad = False

        mask_out = self._set_mask(x_orig)

        c = 1e-1
        c_upper = 10e+10
        c_lower = 0
        
        overall_best_attack = x_orig.clone()
        overall_best_dist = torch.inf
        overall_best_loss = 1e10

        for binary_step in range(self.binary_steps):

            global_step = 0      
            x = x_orig.clone().detach()
            y = x_orig.clone().detach()   

            best_attack = x_orig.clone().detach()
            best_dist = torch.inf
            best_loss = 1e10

            step_size = 1e-2

            for step in range(self.steps):

                y.requires_grad = True
                _, loss = self.loss_fn(x=y, c=c, x_orig=x_orig)
                loss.backward()
                y_grad = y.grad.data * mask_out

                with torch.no_grad():
                    x_new = self.project(x=y-step_size*y_grad, x_orig=x_orig)

                    step_size = (self.step_size - 0) * (1 - global_step / self.steps) ** 0.5 + 0
                    global_step += 1
                    
                    y = x_new + (step / (step + 3)) * (x_new - x)
                    x = x_new
                                    
                    loss_model, loss = self.loss_fn(x=x, c=c, x_orig=x_orig)

                    if self.ver and step % 20 == 0:
                        print(f"Binary Step: {binary_step}, Iter: {step}, Loss: {loss.item()}, L0: {(x - x_orig).norm(p=0)}, Linf: {(x - x_orig).norm(p=torch.inf)}")

                    if self.decision_rule == 'L1':
                        if (x - x_orig).norm(p=1).item() < best_dist and loss_model < best_loss:
                            best_loss = loss_model
                            best_attack = x.clone()
                            best_dist = (x - x_orig).norm(p=1).item()
                    else:
                        raise NotImplementedError

            # Updating c
            if overall_best_dist > best_dist and best_loss < overall_best_loss:
                overall_best_loss = best_loss
                overall_best_dist = best_dist
                overall_best_attack = best_attack.clone()

                c_upper = min(c_upper, c)
                if c_upper < 1e9:
                    c = (c_upper + c_lower) / 2

            else:
                c_lower = max(c_lower, c)
                if c_upper < 1e9:
                    c = (c_lower + c_upper) / 2.0
                else:
                    c *= 10
        
        print(f"Final L0: {(overall_best_attack - x_orig).norm(p=0)}, Linf: {(overall_best_attack - x_orig).norm(p=torch.inf)}")
        return overall_best_attack.detach()


    def project(self, x, x_orig):

        mask_1 = (x - x_orig > self.beta).float()
        mask_2 = ((x - x_orig).abs() <= self.beta).float()
        mask_3 = (x - x_orig < -self.beta).float()

        upper = torch.minimum(x - self.beta, torch.tensor(1.0))
        lower = torch.maximum(x + self.beta, torch.tensor(0.0))
        
        proj_x = mask_1 * upper + mask_2 * x_orig + mask_3 * lower
        return proj_x   
    
    def loss_fn(self, x, c, x_orig):
        
        out = -self.model(x).sum() if not self.targeted else self.model(x).sum()
        l2_dist = ((x - x_orig) ** 2).view(x.shape[0], -1).sum(dim=1)
        l1_dist = ((x - x_orig).abs()).view(x.shape[0], -1).sum(dim=1)

        return out, c * out + l2_dist.sum() + \
                    self.beta * l1_dist.sum()
