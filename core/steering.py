import torch

def compute_steering_vector(pos_states, neg_states):
    vec = pos_states.mean(dim=0) - neg_states.mean(dim=0)
    return vec / vec.norm()


def apply_steering_hook(model, vector, layer=-1, strength=3.0):
    """
    Apply steering vector to model via forward hook.
    
    Works with both GPT-2 (model.transformer.h) and Phi-3 (model.model.layers) architectures.
    """
    # Get device from model
    device = next(model.parameters()).device
    vector = vector.to(device)
    
    # Determine model architecture
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 architecture
        target_module = model.transformer.h[layer]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Phi-3 architecture
        target_module = model.model.layers[layer]
    else:
        raise ValueError("Unknown model architecture. Expected GPT-2 or Phi-3 structure.")
    
    def hook_fn(module, input, output):
        # output is a tuple for transformer layers, modify the hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Broadcast vector: [hidden_dim] -> [1, 1, hidden_dim] to match [batch, seq, hidden_dim]
            steering_addition = strength * vector.unsqueeze(0).unsqueeze(0)
            modified_hidden = hidden_states + steering_addition
            return (modified_hidden,) + output[1:]
        else:
            # Broadcast vector for non-tuple outputs
            steering_addition = strength * vector.unsqueeze(0).unsqueeze(0)
            return output + steering_addition
    
    return target_module.register_forward_hook(hook_fn)
