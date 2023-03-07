import torch
net = torch.load("./CIFAR10_cnn_8_pgd_epoch=16.pt",map_location=torch.device('cpu'))
torch.save(net.state_dict(), "./CIFAR10_cnn_8_pgd_epoch=16.pt")
net = torch.load("./CIFAR10_cnn_8_free_epoch=11.pt",map_location=torch.device('cpu'))
torch.save(net.state_dict(), "./CIFAR10_cnn_8_free_epoch=11.pt")
