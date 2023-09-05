import torch

# n = 10
# i = 5
#
# t = (torch.ones(n)*i).long()
# print(t)
# print(t.shape)
#
# batch = torch.load('./results/DDPM_Uncondtional/9.pt')
# print(batch.shape)
#
# import matplotlib.pyplot as plt
#
# # Your lists of validation and training losses
# validation_losses = [0.2, 0.15, 0.1, 0.08, 0.06]
# training_losses = [0.5, 0.4, 0.3, 0.25, 0.2]
#
# # Create a range of epochs (assuming the same number of epochs for both lists)
# epochs = range(1, len(validation_losses) + 1)
#
# # Plot validation losses in blue and training losses in red
# plt.plot(epochs, validation_losses, 'b', label='Validation Loss')
# plt.plot(epochs, training_losses, 'r', label='Training Loss')
#
# # Add labels and a legend
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Validation and Training Losses')
# plt.legend()
#
# # Save the plot as an image
# plt.savefig('loss_plot.png')
#
# # Display the plot (optional)
# plt.show()

import pickle
categories_indices = pickle.load(open("./pkl_info/slice-mel-512.pkl", "rb"))

print(categories_indices)