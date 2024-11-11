An attempt at understanding and doing the full reproduction given in https://www.youtube.com/watch?v=l8pRSuU81PU
I set up pytorch lightning, and tried several training runs. So far, I could not achieve the same training loss as Karpathy, which may be due to several factors mentioned in the training run 
https://wandb.ai/phyloteam/gpt2/runs/itfh7d3o/overview


- No noticeable speedup in PyTorch 2.5 by compiling the optimization step


# Mistakes
- Gradient clipping at each step instead of accumulating gradient
- Calling optimizer.zero_grad at the start of each training_step instead of only once every gradient_accumulation steps
- Not starting from working code.

# Next steps
- Start from Karpathy's code and iteratively make modifications, ensuring it does not break the learning process