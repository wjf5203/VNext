'''
get backbone from idol
'''

import torch
from backbone import Backbone



def main():

    # backbone = build_backbone(args)
    backbone = Backbone('resnet50', True, True, False)
    # state_dict = torch.load('backbone_weights.pth')['model']
    # backbone.load_state_dict(state_dict,strict=True)

    import numpy as np
    img = np.random.random([8,3,256,256]).astype(np.float64)
    img=torch.from_numpy(img).double()
    out = backbone.forward(img)


   
       
if __name__ == '__main__':
    main()
