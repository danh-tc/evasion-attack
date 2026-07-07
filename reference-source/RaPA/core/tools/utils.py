import os
import numpy as np
from PIL import Image
from mmengine.registry import Registry
import random
import torch
import timm
import torchvision
from torchvision import transforms
import clip
import torch.nn as nn
import sys
class ClipClassifier(nn.Module):
    def __init__(self, model_name):
        super(ClipClassifier, self).__init__()
        if model_name == "CLIP" or 'clip' in model_name:
            model, processor = clip.load("ViT-B/32", device='cuda')
        else:
            raise ValueError(f"Not supported model name. {model_name}")
        self.model = model.float()
        self.model.eval()
        from data.ImageNetClasses import imagenet_classes
        classnames = imagenet_classes
        zeroshot_weights = []
        template = "A photo of a {}."
        with torch.no_grad():
            for classname in classnames:
                texts = template.format(classname) #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    def forward(self, x):
        resize224 = transforms.Resize((224, 224))
        x = resize224(x)
        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * image_features @ self.zeroshot_weights
        return logits
def get_timm_torch_model_checkpoint_path(model_name):
    torch_home = '~/.cache/torch/hub/checkpoints/'
    checkpoint_files = []
    for root, dirs, files in os.walk(os.path.expanduser(torch_home)):
        for file in files:
            if file.endswith('.pth'):
                if model_name in file:
                    return os.path.join(root, file)
    return None

def wrap_model_with_transform(model, transform):
    class wraped_model(nn.modules.Module):
        def __init__(self, model, transform):
            super(wraped_model, self).__init__()
            self.model = model
            transform_list = transform.transforms
            transform_list = [t for t in transform_list if not isinstance(t, transforms.ToTensor)]
            new_transform = transforms.Compose(transform_list)
            self.transform = new_transform
        def forward(self, x):
            x_trs = self.transform(x)
            return self.model(x_trs)
    return wraped_model(model, transform)

def wrap_timm_model(model):
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    model = wrap_model_with_transform(model, transform)
    return model
def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)

def load_model_imagenet(model_name):
    checkpoint_path = get_timm_torch_model_checkpoint_path(model_name)

    if model_name == "ResNet101":
        model = torchvision.models.resnet101(pretrained=True)
    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'ResNet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name == "vgg16":
        model = torchvision.models.vgg16_bn(pretrained=True)
    elif model_name == "vgg19":
        model = torchvision.models.vgg19_bn(pretrained=True)
    elif model_name == "wide_resnet101_2":
        model = torchvision.models.wide_resnet101_2(pretrained=True)
    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True,transform_input=True)
    elif model_name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=True) 
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)

    elif model_name == "mobilenet_v3_large":
        model = torchvision.models.mobilenet.mobilenet_v3_large(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "DenseNet161":
        model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'GoogLeNet':
        model = torchvision.models.googlenet(pretrained=True)
    # timm models
    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
    elif model_name == "inception_resnet_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif model_name == "inception_v3_timm":
        model = timm.create_model("inception_v3", pretrained=True)
    elif model_name == "inception_v4_timm":
        model = timm.create_model("inception_v4", pretrained=True)
    elif model_name == "xception":
        model = timm.create_model("xception", pretrained=True)
    # timm Transformer-based models
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        # model = wrap_timm_model(model)
    elif model_name == "levit_384":
        model = timm.create_model("levit_384", pretrained=True)
        # model = wrap_timm_model(model)

    elif model_name == "convit_base":
        model = timm.create_model("convit_base", pretrained=True)
        # model = wrap_timm_model(model)

    elif model_name == "twins_svt_base":
        model = timm.create_model("twins_svt_base", pretrained=True)
        # model = wrap_timm_model(model)

    elif model_name == "pit":
        model = timm.create_model('pit_s_224', pretrained=True)
        # model = wrap_timm_model(model)

    elif model_name == "ens_adv_inception_resnet_v2":
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        model = wrap_timm_model(model)

    elif "CLIP" or "clip" in model_name:
        model = ClipClassifier(model_name)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    
    return model
# TODO : Add more models in cifar10 and cifar100
def load_model(model_name):
    return load_model_imagenet(model_name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_images(output_dir, adversaries, filenames):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        if filename.endswith('.JPEG'):
            filename = filename.replace('.JPEG', '')
        filename = filename.split("/")[-1]
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename+'.png'))