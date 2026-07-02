#!/bin/bash
for seed in 0 1 2; do
    echo "Running with seed: $seed"
    
    python core/tools/attack.py \
        --config config/RaPA/RaPA-RDI-TI-MI-tar.py \
        --cfg-options \
        work_dir=output/RaPA/tar \
        save_images=True \
        data_loader_cfg.batch_size=32 \
        attacker_cfg.max_iterations=1000 \
        tester_cfg.type=IntervalTester \
        tester_cfg.test_interval=100 \
        attacker_cfg.use_tqdm=True \
        attacker_cfg.drop_name_list=['all'] \
        attacker_cfg.drop_prob=5e-2 \
        attacker_cfg.get_gradient.type=MultiGetGradient \
        attacker_cfg.get_gradient.times=5 \
        source_models_cfg.source_model_names=['ResNet50'] \
        seed=$seed


    python core/tools/attack.py \
        --config config/RaPA/RaPA-RDI-TI-MI-tar.py \
        --cfg-options \
        work_dir=output/RaPA/tar \
        save_images=True \
        data_loader_cfg.batch_size=32 \
        attacker_cfg.max_iterations=1000 \
        tester_cfg.type=IntervalTester \
        tester_cfg.test_interval=100 \
        attacker_cfg.use_tqdm=True \
        attacker_cfg.drop_name_list=['all'] \
        attacker_cfg.drop_prob=4e-2 \
        attacker_cfg.get_gradient.type=MultiGetGradient \
        attacker_cfg.get_gradient.times=5 \
        source_models_cfg.source_model_names=['DenseNet121'] \
        seed=$seed

    python core/tools/attack.py \
        --config config/RaPA/RaPA-RDI-TI-MI-tar.py \
        --cfg-options \
        work_dir=output/RaPA/tar \
        save_images=True \
        data_loader_cfg.batch_size=32 \
        attacker_cfg.max_iterations=1000 \
        tester_cfg.type=IntervalTester \
        tester_cfg.test_interval=100 \
        attacker_cfg.use_tqdm=True \
        attacker_cfg.drop_name_list=['all'] \
        attacker_cfg.drop_prob=2e-2 \
        attacker_cfg.get_gradient.type=MultiGetGradient \
        attacker_cfg.get_gradient.times=5 \
        source_models_cfg.source_model_names=['inception_v3'] \
        seed=$seed
done