###########################################
# Utils Functions for Training and Testing
###########################################
import torch
import torch.nn as nn

def make_predictions(model: nn.Module, data: list, device: torch.device):
    # prediction probabilities list
    pred_probs = []
    # setting the model to eval mode
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # adding an extra dimension to sample and putting it to device
            sample = torch.unsqueeze(sample, dim=0).to(device)
            # forward pass to model and getting logits
            y_logits = model(sample)
            # squeezing dimension and getting pred labels
            pred_prob = torch.softmax(y_logits.squeeze(), dim=0)
        
        # adding pred prob to list and setting it to cpu
        pred_probs.append(pred_prob.cpu())
    
    # returning stacked list
    return torch.stack(pred_probs)

