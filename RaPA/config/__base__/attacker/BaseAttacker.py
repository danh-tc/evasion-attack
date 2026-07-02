save_images=True
test_when_generate=True
seed=0

attacker_cfg=dict(
    type="BaseAttacker",
    epsilon=16/255,
    alpha=2/255,
    max_iterations=1000,
    num_images=1000,    
)