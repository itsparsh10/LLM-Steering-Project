import torch

def get_hidden_states(model, tokenizer, texts, layer=-1):
    states = []
    
    # Get device from model
    device = next(model.parameters()).device

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        states.append(outputs.hidden_states[layer][0, -1])

    return torch.stack(states)
