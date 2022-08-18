import torch.nn as nn


class MyEfficientNetB(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self, eff_net_backbone, num_categories, dropout=0.2) -> None:
        super().__init__()

        self.num_categories = num_categories
        self.backbone = eff_net_backbone

        # Disable Backbone's Classification Layer
        embedding_size = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.my_classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size//2),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(embedding_size//2, embedding_size//4),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(embedding_size//4, self.num_categories),
        )
        # Adding Softmax Layer
        # self.softmax = nn.Softmax(dim=1)

        # self.act_classification = nn.Sigmoid()

    def forward(self, _in):
        """
            Forwarding input to output
        """
        # print("******** Before fc: {}".format(out.shape))
        out = self.backbone(_in)

        out = self.my_classifier(out)
        # out = self.softmax(out)

        return out

    def freeze_backbone(self):
        """
            Freezing Backbone Network
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
            Unfreezing Backbone Network
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_layers(self, layers_index):
        """
            Unfreezing Some backbone layers
        """
        for layer in layers_index:
            for param in self.backbone.features[layer].parameters():
                param.requires_grad = True
