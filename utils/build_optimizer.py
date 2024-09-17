import math
import torch
from torch.optim import Adam, Adamax, Optimizer
from .logger import LOGGER






def build_optimizer(model, args, checkpoint_optim):

    vision_clip = 'vision_encoder_type' in args.model_cfg and 'clip' in args.model_cfg.vision_encoder_type

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
   
    basic_params = []
    basic_params_name = []
    basic_params_no_decay = []
    clip_params_visual = []
    clip_params_name_visual = []
    clip_params_no_decay_visual = []
    clip_params_text = []
    clip_params_name_text = []
    clip_params_no_decay_text = []
    new_params = []
    new_params_name = []
    new_params_no_decay = []


    for k, v in model.named_parameters():
        if any(nd in k for nd in args.run_cfg.new_params_name) and not any(nd in k for nd in no_decay):
            new_params.append(v)
            new_params_name.append(k)
        elif any(nd in k for nd in args.run_cfg.new_params_name) and any(nd in k for nd in no_decay):
            new_params_no_decay.append(v) 
            new_params_name.append(k)
        elif  vision_clip  and 'visual' in k and not any(nd in k for nd in no_decay):
            clip_params_visual.append(v)
            clip_params_name_visual.append(k)
        elif vision_clip  and  'visual' in k and  any(nd in k for nd in no_decay):
            clip_params_no_decay_visual.append(v)
            clip_params_name_visual.append(k)
     
        
        elif not any(nd in k for nd in no_decay):
            basic_params.append(v)
            basic_params_name.append(k)
        elif any(nd in k for nd in no_decay):
            basic_params_no_decay.append(v)
            basic_params_name.append(k)

    # print(new_params)
    optimizer_grouped_parameters = [
        {'params': basic_params, 'weight_decay': args.run_cfg.weight_decay, 'lr': args.run_cfg.learning_rate},
        {'params': basic_params_no_decay, 'weight_decay': 0.0, 'lr': args.run_cfg.learning_rate},
        {'params': new_params, 'weight_decay': args.run_cfg.weight_decay, 'lr': args.run_cfg.new_lr},
        {'params': new_params_no_decay, 'weight_decay': 0.0, 'lr': args.run_cfg.new_lr},
        {'params': clip_params_visual, 'weight_decay': args.run_cfg.weight_decay, 'lr': args.run_cfg.clip_lr},
        {'params': clip_params_no_decay_visual, 'weight_decay': 0.0, 'lr': args.run_cfg.clip_lr},
    ]

    # print(basic_params_name)
    # print(clip_params_visual)
    # currently Adam only
    if args.run_cfg.optim == 'adam':
        OptimCls = Adam
    elif args.run_cfg.optim == 'adamax':
        OptimCls = Adamax
    elif args.run_cfg.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')

    for i in optimizer_grouped_parameters:
        i['init_lr'] = i['lr']
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=args.run_cfg.learning_rate, betas=args.run_cfg.betas)

    optimizer.new_params_name = new_params_name
    optimizer.new_lr = args.run_cfg.new_lr
    optimizer.basic_lr = args.run_cfg.learning_rate
    optimizer.clip_lr_visual = args.run_cfg.clip_lr
    optimizer.clip_lr_visual_len = len(clip_params_visual)

    optimizer.zero_grad()

    if checkpoint_optim:
        optimizer.load_state_dict(checkpoint_optim)
        del(checkpoint_optim)

    LOGGER.info('==='*6+'learning_rate_settings'+'==='*6+'\n')
    LOGGER.info(f"  basic_lr : {optimizer.basic_lr}")
    LOGGER.info(f"  clip_lr_visual : {optimizer.clip_lr_visual}")
    LOGGER.info(f"  clip_lr_visual_len : {optimizer.clip_lr_visual_len}")
    LOGGER.info(f"  new_lr : {optimizer.new_lr}")
    LOGGER.info(f"  new_params_name: {optimizer.new_params_name}")

    return  optimizer





class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2).
            Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias
            in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - "
                             "should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - "
                             "should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - "
                             "should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse '
                        'gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = (step_size * math.sqrt(bias_correction2)
                                 / bias_correction1)

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is
                # *not* the correct way of using L2 regularization/weight decay
                # with Adam, since that will interact with the m and v
                # parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't
                # interact with the m/v parameters. This is equivalent to
                # adding the square of the weights to the loss with plain
                # (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss