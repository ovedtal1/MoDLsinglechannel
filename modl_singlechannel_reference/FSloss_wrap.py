import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from fastmri.models import Unet
from recon_net import ReconNet
import torchvision
from torch.nn import functional as F
from torchvision import models, transforms

class FSloss(nn.Module):
    def __init__(self, net, epsilon=1e-6):
        super().__init__()
        self.device = 'cuda:0'

        # ViT layer
        self.feature_net = models.resnet18(weights='DEFAULT').to(self.device).requires_grad_(False)
        self.recon_net_ref = ReconNet(net).to(self.device)
        # Load weights
        cp = torch.load('./lsdir-2x+hq50k_vit_epoch_60.pt', map_location=self.device)
        self.recon_net.load_state_dict(cp['model_state_dict'])
        self.recon_net_ref.load_state_dict(cp['model_state_dict'])


    def forward(self, img,ref): #,ref
        # Norm
        #print("Current value of param1 during forward:", self.param1)
        #print("Current value of param2 during forward:", self.param2)
        in_pad, wpad, hpad = self.recon_net.pad(img)
        ref_pad, wpad, hpad = self.recon_net.pad(ref)
        input_norm,mean,std = self.recon_net.norm(in_pad.float())
        ref_norm,mean_ref,std_ref = self.recon_net.norm(ref_pad.float())
        #print("Weights of the Conv1 layer:")
        #print(self.conv1.weight)        
        # Feature extract
        features = self.recon_net.net.forward_features(input_norm)#.permute(0,2,1)
        #features = self.conv1_acq1(features)
        
        
        features_ref = self.recon_net_ref.net.forward_features(ref_norm)#.permute(0,2,1)
        #features_ref = self.conv1_acq1(features_ref)
        """
        acq1 = self.lrelu(self.conv1_acq1(features))
        acq2 = self.lrelu(self.conv1_acq2(features_ref))

        # Compute similarity weightings
        similarity = torch.sigmoid(acq1 - acq2)
        
        # Element-wise operations (fusion process)
        features_ref = features * similarity + features_ref * (1 - similarity)

        # mRCAB processing
        features_ref = self.mrcab(features_ref)
        """



        #features = (features + features_ref)/2 
        #print(f'fetures shape: {features.shape}')
        # Fusion
        
        batch_size, num_channels, height = features.shape
        features_flat = features.reshape(batch_size, num_channels, -1)
        features_ref_flat = features_ref.reshape(batch_size, num_channels, -1)       
        
        # Reshape params to match the dimensions
        param1_expanded = self.param1.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        param2_expanded = self.param2.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        # Expand params to match the flattened tensor dimensions
        param1_expanded = param1_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        param2_expanded = param2_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        # Calculate weighted sum

        weighted_sum = (param1_expanded * features_flat + param2_expanded * features_ref_flat)

        
        # Calculate normalization factor
        normalization_factor = param1_expanded + param2_expanded + self.epsilon
        
        # Normalize
        features_comb = weighted_sum / normalization_factor
        #features_comb = self.rg_fusion(features_comb.squeeze(0)).unsqueeze(0)     
        
        # Reshape back to [1, 416, 1024] - High Resolution
        #features_comb = features_comb.reshape(features_flat.shape[0], 416, 1024)
        # Low Resolution
        features_comb = features_comb.reshape(features_flat.shape[0], 198, 1024)
        
        #features_comb = self.fuser(features,features_ref)
        #print(f'features_comb: {features_comb.shape}')
        
        # Recon Head
        head_out = self.recon_net.net.head(features_comb)
        
        # High Resolution
        #head_out_img = self.recon_net.net.seq2img(head_out, (260, 160))
        # Low Resolution 
        head_out_img = self.recon_net.net.seq2img(head_out, (180, 110))


        # un-norm
        merged = self.recon_net.unnorm(head_out_img, mean, std) 

        # un-pad 
        im_out = self.recon_net.unpad(merged,wpad,hpad)
        
        return im_out
    




class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        #self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.l1_loss(input_feats, target_feats, reduction=self.reduction)
    


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super(ResNet18Backbone, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
        )

    def forward(self, x):
        return self.feature_extractor(x)
    

class FeatureEmbedding(nn.Module):
    def __init__(self, backbone, output_dim=128):
        super(FeatureEmbedding, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, output_dim)  # Adjust 512 if needed based on feature size

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        embedding = self.fc(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)  # L2 normalization
        return embedding

def contrastive_loss(embedding, memory_bank):
    positive = embedding.dot(memory_bank[0])  # Positive sample (or query itself)
    negatives = embedding.dot(memory_bank[1:])  # Negatives
    loss = -torch.log(torch.exp(positive) / (torch.exp(positive) + torch.sum(torch.exp(negatives))))
    return loss


"""
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[2, 3], style_layers=[]):
        #[0, 1, 2, 3]
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.mse_loss(gram_x, gram_y)
        return loss
"""
class VGGPerceptualLoss(nn.Module):
    DEFAULT_FEATURE_LAYERS = []#[0, 1, 2, 3]
    IMAGENET_RESIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGENET_SHAPE = (1, 3, 1, 1)

    def __init__(self, resize=True, feature_layers=None, style_layers=None):
        super().__init__()
        self.resize = resize
        self.feature_layers = feature_layers or self.DEFAULT_FEATURE_LAYERS
        self.style_layers = style_layers or [2,3]
        features = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            features[:4].eval(),
            features[4:9].eval(),
            features[9:16].eval(),
            features[16:23].eval(),
        ])
        #for param in self.parameters():
            #param.requires_grad = False
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(self.IMAGENET_SHAPE))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(self.IMAGENET_SHAPE))

    def _transform(self, tensor):
        if tensor.shape != self.IMAGENET_SHAPE:
            tensor = tensor.repeat(self.IMAGENET_SHAPE)
        """
        print(tensor.shape)
        print(self.mean.shape)
        print(self.std.shape)
        """
        tensor = (tensor - self.mean) / self.std
        if self.resize:
            tensor = nn.functional.interpolate(tensor, mode='bilinear', size=self.IMAGENET_RESIZE, align_corners=False)
        return tensor

    def _calculate_gram(self, tensor):
        act = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return act @ act.permute(0, 2, 1)

    def forward(self, output, target):
        output, target = self._transform(output), self._transform(target)
        loss = 0.
        for i, block in enumerate(self.blocks):
            output, target = block(output), block(target)
            if i in self.feature_layers:
                loss += nn.functional.l1_loss(output, target)
            if i in self.style_layers:
                gram_output, gram_target = self._calculate_gram(output), self._calculate_gram(target)
                loss += nn.functional.l1_loss(gram_output, gram_target)
        return loss
    

