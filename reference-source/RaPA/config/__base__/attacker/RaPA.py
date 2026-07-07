_base_ =["./BaseAttacker.py"]

attacker_cfg=dict(
    type="DropConnect",
    drop_prob=1e-2, # 1e-2 vit, 5e-2 resnet50, 3e-2 clip, 4e-2 dense121, 2e-2 incv3
    drop_name_list=["all"],
    type_list=["Linear", "Normalization"],
)