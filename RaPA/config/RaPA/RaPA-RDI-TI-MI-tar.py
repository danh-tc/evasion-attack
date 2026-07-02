_base_ = [
    '../__base__/dataset/ImageNetCompetition.py',
    '../__base__/tester/targeted.py',
    '../__base__/input_transform/RDI.py',
    '../__base__/loss_fn/Logit.py',
    '../__base__/get_gradient/BaseGetGradient.py',
    '../__base__/update_gradient/TI-MI.py',
    '../__base__/attacker/RaPA.py'
]
name="RaPA-RDI-TI-MI-tar"
source_models_cfg=dict(
    source_model_names=[
        'ResNet50'
    ],
)
attacker_cfg=dict(
    input_transform=_base_.input_transform,
    loss_fn=_base_.loss_fn,
    get_gradient=_base_.get_gradient,
    update_gradient=_base_.update_gradient,
)

