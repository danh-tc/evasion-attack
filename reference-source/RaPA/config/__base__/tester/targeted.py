tester_cfg=dict(
    type="BaseTester",
    is_targeted=True,
    test_model_names=[
        'ResNet18', 'ResNet50', 'vgg16', 'inception_v3', 'efficientnet_b0',
        'DenseNet121', 'mobilenet_v2', 'inception_resnet_v2',
        'inception_v4_timm', 'xception', 'vit_base_patch16_224',
        'levit_384', 'convit_base', 'twins_svt_base', 'pit', 'CLIP'
    ],
    need_resize_names=[
        'vit_base_patch16_224', 'levit_384', 'convit_base', 'twins_svt_base', 'pit'
    ],
)
