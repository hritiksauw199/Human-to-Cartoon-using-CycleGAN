import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("./checkpoints/train/best_checkpoint.pth", map_location=device)

#best_epoch = checkpoint['epoch']
#G_loss_best = checkpoint['G_loss']
#D_loss_best = checkpoint['D_loss']


print(checkpoint['epoch'])
#print(G_loss_best)
#print(D_loss_best)