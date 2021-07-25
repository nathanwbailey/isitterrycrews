import torch
from torch import nn
import torchvision
from senet import se_resnet50
import pytorch_model_summary as pms

class TripletNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # layer_list = list(se_resnet50().layer0)
        # layer_list.extend(list(se_resnet50().layer1))
        # layer_list.append(se_resnet50().layer2[0])
        # layer_list.extend([se_resnet50().layer2[1].conv1, se_resnet50().layer2[1].bn1, se_resnet50().layer2[1].conv2, se_resnet50().layer2[1].bn2, se_resnet50().layer2[1].conv3, se_resnet50().layer2[1].bn3, se_resnet50().layer2[1].relu, se_resnet50().layer2[1].se_module.avg_pool])
        # self.backbone = nn.Sequential(*layer_list)
        self.backbone = se_resnet50()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.relu = nn.ELU()
        self.backbone.last_linear = nn.Linear(2048, 512)
        self.backbone.last_linear.weight.requires_grad = True
        self.backbone.last_linear.bias.requires_grad = True
        self.feature = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.6)
        )

        self.classification = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.relu((self.backbone(x)))
        x = x.reshape(x.size(0), -1)
        x = self.feature(x)
        y = self.classification(x)
        return x, y
    
    # def forward(self, x1, x2, x3):
    #     anchor_emb, anchor_class = self.forward_once(x1)
    #     pos_emb, pos_class = self.forward_once(x2)
    #     neg_emb, neg_class = self.forward_once(x3)
    #     return anchor_emb, anchor_class, pos_emb, pos_class, neg_emb, neg_class
    

# model = TripletNetwork(776)
# model = se_resnet50()
# pms.summary(model, torch.zeros((1, 3, 224, 224)), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
# pms.summary(model, torch.zeros((1, 3, 224, 224)), torch.zeros((1, 3, 224, 224)), torch.zeros((1, 3, 224, 224)), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)




