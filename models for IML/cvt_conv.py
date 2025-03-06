import torch
a = torch.load('upernet_convnext_base_22k_640x640.pth', map_location='cpu')['state_dict']
for hd in ('decode_head', 'auxiliary_head'):
    for nm in ('weight', 'bias'):
        a['%s.conv_seg.%s'%(hd, nm)] = a['%s.conv_seg.%s'%(hd, nm)][:2]
torch.save({'state_dict': a}, 'convnext_ade.pth')
