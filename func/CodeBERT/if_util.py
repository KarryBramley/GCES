# from flax import optim
# from flax.struct import dataclass as flax_dataclass
# from flax.training import checkpoints
# from flax import linen as nn
# from jax import jit, random
# from jax import numpy as jnp
# from jax.scipy.special import expit

import time
import torch
import torch.autograd as autograd
import numpy as np


# class FlaxModel(nn.Module):
#     def setup(self, encoder, config, tokenizer, args):
#         self.encoder = encoder
#         self.config = config
#         self.tokenizer = tokenizer
#         self.args = args
#         self.weight = 1
    
#     @nn.compact
#     def __call__(self, input_ids=None,labels=None,relay=None):
#         outputs=self.encoder(input_ids, attention_mask = (input_ids != 1))[0]
#         logits=outputs
#         prob=expit(logits)
#         return prob
    
# def get_apply_fn_test(model):
#   def apply_fn_test(params, model_state, x):
#     vs = {'params': params, **model_state}
#     logits = model.apply(vs, x, train=False, mutable=False)
#     return logits
#   return apply_fn_test

# @flax_dataclass
# class TrainState:
#   optim: optim.Optimizer
#   model: Any


# def create_train_state(args, model):
#   @jit
#   def init(*args):
#     return model.init(*args)
#   key, input = random.PRNGKey(args.model_seed), jnp.ones((1, *args.image_shape), model.dtype)
#   model_state, params = init(key, input).pop('params')

#   opt = optim.Momentum(args.lr, args.beta, args.weight_decay, args.nesterov).create(params)
#   train_state = TrainState(optim=opt, model=model_state)
#   return train_state

# def get_train_state(args, model):
#   time_start = time.time()
#   print('get train state... ', end='')
#   state = create_train_state(args, model)
#   if args.load_dir:
#     print(f'load from {args.load_dir}/ckpts/checkpoint_{args.ckpt}... ', end='')
#     state = checkpoints.restore_checkpoint(args.load_dir + '/ckpts', state, args.ckpt)
#   print(f'{int(time.time() - time_start)}s')
#   return state, args


def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def hv(loss, model_params, v): # according to pytorch issue #24004
    # import time
    # s = time.time()
    grad = autograd.grad(loss[0], model_params, create_graph=True, retain_graph=True)
    # e1 = time.time()
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    # e2 = time.time()
    # print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
    return Hv


######## LiSSA ########

def get_inverse_hvp_lissa(v, model, device, param_influence, train_loader, damping, num_samples, recursion_depth, scale=1e4):
    ihvp = None
    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in range(recursion_depth):
            try:
                input_ids, labels = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                input_ids, labels = next(lissa_data_iterator)
            input_ids = input_ids.to(device)
            labels = labels.to(device)  
            model.zero_grad()
            train_loss = model(input_ids, labels)
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            if (j % 200 == 0) or (j == recursion_depth - 1):
                print("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]

    # return ihvp
    
    # print("ihvp维度为:")
    # for h in ihvp:
    #     print(h.shape)

    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    # 查看下ihvp的维度
    # return_ihvp维度为:
    # torch.Size([124646401])
    #              43118593
    return return_ihvp
