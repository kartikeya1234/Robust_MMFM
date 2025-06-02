import torch


class Attack(object):
    '''
    Root class for all adversarial attack classes.
    '''

    def __init__(self, model, targeted=False, img_range=(0, 1)):
        self.model = model
        self.device = 'cuda:0'
        self.targeted = targeted
        self.img_range = img_range

    def __repr__(self):
        return str(self.__dict__)

    def to(self, device):
        self.model.to(device)
        self.device = device