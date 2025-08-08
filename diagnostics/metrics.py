import torch

"""
Contains the isolated computations done for metrics or hooks.
Removes the logic from the file itself. 
"""

def flatten(params):
    flats = [p.reshape(-1) for p in params] #removed an if p.requires grad. 
   
    return torch.cat(flats)
def unflatten_like(vec, params):
    outs, i = [], 0
    for p in params:
        if not p.requires_grad: 
            outs.append(None); 
            continue
        n = p.numel()
        outs.append(vec[i:i+n].view_as(p)); i += n
    return outs

@torch.no_grad()
def params_vector(model):
    return flatten([p for p in model.parameters()])

def hvp(model, batch, v_flat, epoch, create_graph = False):
    """
    Apply hessian H to vector v. 
    Hv = \nabla_{\theta}(\nabla_{\theta}L(\theta)\cdot v)

    Call with model.eval(), and a FIXED batch. 
    """
    model.zero_grad()

    loss_dict = model.compute_loss(batch, epoch) # note: If model loss has weight decay, it's not in th ehessian. 
    params = [p for p in model.parameters() if p.requires_grad]

    g = torch.autograd.grad(loss_dict["loss"], params, create_graph = True)#need create_graph = True
    

    g_flat = flatten(g)

    s = (g_flat * v_flat).sum()

    Hv = torch.autograd.grad(s, params, retain_graph = False, create_graph = create_graph)
    
    #print(Hv.shape)
    Hv_flat = flatten(Hv).detach()# keep graph only if you need higher order. 
  
    return Hv_flat


def lanczos(Hv_op, n, m, seed, device):
    g = torch.Generator(device = device).manual_seed(seed)#is this already set
    v0 = torch.randn(n, generator = g, device = device)
    v0 = v0/v0.norm()
    Vs = [v0]
    alpha, beta = [], []
    w = Hv_op(Vs[-1])
    
    a0 = torch.dot(Vs[-1], w).item() # computes v^T H v

    w = w - a0 * Vs[-1] #
    alpha.append(a0)

    for j in range(1, m):
        b = w.norm().item()
        beta.append(b)
        if b < 1e-12:
            break
        v = w / (b + 1e-12)
        Vs.append(v)
        w = Hv_op(v) - beta[-1] * Vs[-2]
        a = torch.dot(v, w).item()
        w = w - a * v
        alpha.append(a)

    T = torch.zeros(len(alpha), len(alpha), device=device)
    T = torch.diag(torch.tensor(alpha, device = device))

    if beta:
        b = torch.tensor(beta, device=device)
        T[:-1,1:] += torch.diag(b)
        T[1:,:-1] += torch.diag(b)

    evals, evecs = torch.linalg.eigvalsh(T), None
    return evals  # approximate spectrum along the Krylov subspace

def CKA(X, Y):
    """
    Centered Kernel Alignment - similarity between two sets of representations. 
    For nonlinear, can use RBF kernels. 

    If it's close to one, X and y are essentially the same. More stable than raw cosine similairty. 
    Also looks at

    CKA = frobenius inner product normalized by its values. It's literally cosine similarity between two normalized versions of the matrices. 
    """
    n = X.shape[0]
    assert(Y.shape[0] == n)
    Kx = X@ X.T
    Ky = Y@ Y.T
    ones = torch.ones((n,1))
    H = torch.identity(n) - 1/n * ones@ones.T #i've used this one before? 
    Kxh = H@Kx@H
    Kyh = H@Ky @ H
    #Note: All of the above is the same as basically just subtrac cting the mean from each column. 
    CKA = torch.trace(Kyh.T@Kxh)/ (torch.trace(Kxh.T@Kxh)*torch.trace(Kyh.T@Kyh))
    return CKA
    

