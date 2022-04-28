from fastai.vision.all import *

class classification_model(nn.Module):
    def __init__(self, n_classes, embedding_dimension = 512, dropout = 0.4):
        
        super(classification_model, self).__init__()# Define the backbone
        self.backbone = create_body(resnet50)
        self.pool = AdaptiveConcatPool2d(1)
        
        # Define the layernorm layer here
        ftrs = num_features_model(self.backbone) * 2
        self.downscale = nn.Linear(ftrs, embedding_dimension)
        self.dropout = nn.Dropout(dropout)
        
        # Define a linear layer at the end
        self.fc = nn.Linear(embedding_dimension, n_classes, bias = False) 
    
    def forward(self, x):
        # Pass through bakcbone and get fv
        fmaps = self.pool(self.backbone(x))
        global_feature = fmaps.view(fmaps.size(0), -1)
        
        
        # Layernorm the feature vector
        lnormed_embedding = self.dropout(F.layer_norm(global_feature, [global_feature.size(-1)]))
        feature = F.normalize(self.downscale(lnormed_embedding), dim = -1)
        
        # FC Layer without bias to reduce dimensionality from embedding to output classes space
        weight_norm = F.normalize(self.fc.weight, dim = -1)
        prediction_logits = feature.matmul(weight_norm.T)
        
        return prediction_logits
    
    def get_fv(self, x):
        # Pass through bakcbone and get fv
        fmaps = self.pool(self.backbone(x))
        global_feature = fmaps.view(fmaps.size(0), -1)
        
        # Layernorm the feature vector
        lnormed_embedding = F.layer_norm(global_feature, [global_feature.size(-1)])
        
        # Project down the layernormed embedding
        feature = F.normalize(self.downscale(lnormed_embedding), dim = -1)
        
        return feature
        

class NormSoftmaxLoss(nn.Module):
    """
    Apply temperature scaling on logits before computing the cross-entropy loss.
    """
    def __init__(self, temperature=0.05):
        super(NormSoftmaxLoss, self).__init__() 
        self.temperature = temperature
        self.loss_fn = CrossEntropyLossFlat()
 
    def forward(self, prediction_logits, instance_targets):
        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return loss

class NormSoftmaxFocalLoss(nn.Module):
    """
    Apply temperature scaling on logits before computing the cross-entropy loss.
    """
    def __init__(self, temperature=0.05, gamma = 2):
        super(NormSoftmaxFocalLoss, self).__init__() 
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss(reduction = "none")
        self.gamma = gamma
 
    def forward(self, prediction_logits, instance_targets):
        probas = prediction_logits.softmax(dim = -1)
        pt = probas[range(instance_targets.shape[0]), instance_targets]
        loss = ((1 - pt) ** self.gamma) * (self.loss_fn(prediction_logits / self.temperature, instance_targets))        
        return loss.mean()


def effnet_splitter(m):
    return L(m.backbone, m.downscale, m.fc).map(params)