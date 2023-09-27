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
import json

categories_indices = pickle.load(open("./db_metadata/nesmdb/nesmdb_categories.pkl", "rb"))

emotions = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}

categories_indices["emotions"] = emotions

file = open('./db_metadata/nesmdb/nesmdb_categories.pkl', 'wb')
pickle.dump(categories_indices, file)
file.close()

y = json.dumps(categories_indices, indent=4)
file_json = open('./db_metadata/nesmdb/nesmdb_categories.json', 'w')
file_json.write(y)
file_json.close()


print(categories_indices)