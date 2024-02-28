import torch
import numpy as np
import torch.nn as nn

#
# Implements a version of the NeuralODE adjoint optimisation algorithm, with the Adaptive Checkpoint Adjoint method
#
# Original NODE : https://arxiv.org/abs/1806.07366
# ACA version : https://arxiv.org/abs/2006.02493

#
# NB : This code is heavily based on the torch_ACA package (https://github.com/juntang-zhuang/torch_ACA)


# Used for float comparisons
tiny = 1e-8


class odeint_ACA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z0, func, options, *params):

        # Saving parameters for backward()
        ctx.func = func
        # ctx.flat_params = flat_params
        with torch.no_grad():
            ctx.save_for_backward(*params)
        ctx.options = options
        ctx.nSteps = options["nSteps"]
        ctx.dt = options["dt"]
        t = options["t0"]
        # Device used
        argdev = z0.device

        argshape = z0.shape
        argdtype = z0.dtype

        # Simulation
        with torch.no_grad():
            values = [z0]
            alltimes = [t]
            for i in range(ctx.nSteps):
                val = func.step(t, values[-1], dt=ctx.dt)
                t += ctx.dt
                values.append(val)
                alltimes.append(t)
        # Retrieving the time stamps selected by the solver
        ctx.alltimes = alltimes

        ctx.allstates = torch.stack(values)
        evaluations = ctx.allstates[options["eval_idx"]]

        return evaluations

    @staticmethod
    def backward(ctx, *grad_y):
        # This function implements the adjoint gradient estimation method for NODEs

        # grad_output holds the gradient of the loss w.r.t. each evaluation step
        grad_output = grad_y[0]
        # print(grad_output)

        # h is the value of the forward time step
        h = ctx.dt

        # Retrieving the time mesh and the corresponding states created in forward()
        allstates = ctx.allstates
        time_mesh = ctx.alltimes

        # f_params holds the NODE parameters for which a gradient will be computed
        # f_params = []
        # for p in ctx.func.parameters():
        #    if p.requires_grad:
        #        f_params.append(p)
        params = ctx.saved_tensors

        # The last step of the time mesh is an evaluation step, thus the adjoint state which corresponds to the
        # gradient of the loss w.r.t. the evaluation states is initialised with the gradient corresponding to
        # the last evaluation time.

        adjoint_state = grad_output[-1]
        i_ev = -2

        out2 = None

        # The adjoint state as well as the parameters' gradient are integrated backwards in time.
        # Following the Adaptive Checkpoint Adjoint method, the time steps and corresponding states of the forward
        # integration are re-used by going backwards in the time mesh.
        for i in range(len(time_mesh), 0, -1):

            # Backward Integrating the adjoint state and the parameters' gradient between time i and i-1
            z_var = torch.autograd.Variable(allstates[i - 1], requires_grad=True)

            with torch.enable_grad():
                # Taking a step with the NODE function to build a graph which will be differentiated
                # so as to integrate the adjoint state and the parameters' gradient
                y = ctx.func.step(time_mesh[i - 1], z_var, h)

                # Computing the increment to the parameters' gradient corresponding to the current time step
                param_inc = torch.autograd.grad(
                    y, params, adjoint_state, retain_graph=True
                )

                # The following line corresponds to an integration step of the adjoint state
                adjoint_state = torch.autograd.grad(y, z_var, adjoint_state)[0]

            # incrementing the parameters' grad
            if out2 is None:
                out2 = param_inc
            else:
                for _1, _2 in zip([*out2], [*param_inc]):
                    _1 += _2

            # When reaching an evaluation step, the adjoint state is incremented with the gradient of the corresponding
            # evaluation step
            next_i = i - 1
            if next_i in ctx.options["eval_idx"] and i != len(time_mesh):
                adjoint_state += grad_output[i_ev]
                i_ev = i_ev - 1
        # Returning the gradient value for each forward() input
        out = tuple([adjoint_state] + [None, None]) + out2

        return out


def odesolve_adjoint(z0, func, options):
    # Main function to be called to integrate the NODE

    # z0 : (tensor) Initial state of the NODE
    # func : (torch Module) Derivative of the NODE
    # options : (dict) Dictionary of solver options, should at least have a

    # The parameters for which a gradient should be computed are passed as a flat list of tensors to the forward function
    # The gradient returned by backward() will take the same shape.
    # flat_params = flatten_grad_params(func.parameters())
    params = find_parameters(func)

    # Forward integrating the NODE and returning the state at each evaluation step
    zs = odeint_ACA.apply(z0, func, options, *params)
    return zs


def flatten_grad_params(params):
    # Parameters for which a grad is required are flattened and returned as a list
    flat_params = []
    for p in params:
        if p.requires_grad:
            flat_params.append(p.contiguous().view(-1))

    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


def flatten_params(params):
    # values in the params tuple are flattened and returned as a list
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


def get_integration_options(n0, n1, dt, substeps):

    sub_dt = float(dt / substeps)
    nSteps = substeps * (n1 - n0)

    integration_options = {
        "t0": n0 * dt,
        "dt": sub_dt,
        "nSteps": nSteps,
        "eval_idx": np.arange(0, nSteps + 1, substeps),
    }

    return integration_options


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:

        return [r for r in module.parameters() if r.requires_grad]
