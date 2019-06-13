from torch import optim
    
def get_optimizer(model, model_param):
    learning_rate = float(model_param['learning_rate'])
    return  optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                       rho=0.95, eps=1e-06, weight_decay=0)