import copy
import torch
def FedAvg(params):
    """
    Average the paramters from each client to update the global model
    :param params: list of paramters from each client's model
    :return global_params: average of paramters from each client
    """
    global_params = copy.deepcopy(params[0])
    for key in global_params.keys():
        idx = 0
        for param in params[1:]:
            # global_params[key] += (param[key] * dataset_len[idx])
            global_params[key] += param[key]
            idx+=1
        global_params[key] = torch.div(global_params[key], len(params))

    return global_params
