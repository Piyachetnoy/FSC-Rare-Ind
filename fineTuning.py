# https://christiangrech.medium.com/fine-tuning-your-own-custom-pytorch-model-e3aeacd2a819
import torch

# Load your custom model
model = get_model()

# Load the parameters from your saved .pth file
model.load_state_dict(torch.load(path_to_your_pth_file))

# Define the optimizer including the Learning rate and Momentum
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

finetune_epochs = 10  # Number of epochs for fine-tuning
for epoch in range(finetune_epochs):
    # Train your model
    train_model(model)
    # Validate your model
    validate_model(model)

# Saved fine-tuned model
torch.save(model.state_dict(), 'finetuned_model.pth')