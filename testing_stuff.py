import torch

n = 10
i = 5

t = (torch.ones(n)*i).long()
print(t)
print(t.shape)

batch = torch.load('./results/DDPM_Uncondtional/9.pt')
print(batch.shape)