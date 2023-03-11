import torch
import sys

if (len(sys.argv) == 2):
    save_path = sys.argv[1]
    net = torch.load("./"+save_path, map_location=torch.device('cpu'))
    torch.save(net.state_dict(), "./"+save_path)
    print("state_dict change successful")
    # net = torch.load("./CIFAR10_cnn_8_free_epoch=11.pt",map_location=torch.device('cpu'))
    # torch.save(net.state_dict(), "./CIFAR10_cnn_8_free_epoch=11.pt")
else:
    print("usage : python change_state_dict.py <save_path.pt>")
