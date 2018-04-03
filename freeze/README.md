
save as ckpt
```
python saveas_model.py \
    --checkpoint_path ../models/east_icdar2015_resnet_v1_50_rbox \
    --output_model_dir ../models/east_icdar2015_resnet_v1_50_rbox_ckpt \
    --model_name model.ckpt
```

ckpt2pb
```
python freeze.py \
    --model_dir ../models/east_icdar2015_resnet_v1_50_rbox_ckpt \
    --output_graph ../models/east_icdar2015_resnet_v1_50_rbox_ckpt/freeze_model.pb \
    --output_node_names feature_fusion/Conv_7/Sigmoid,feature_fusion/concat_3
```

测试
```
python load.py \
    --frozen_model_filename ../models/east_icdar2015_resnet_v1_50_rbox_ckpt/freeze_model.pb \
    --image ../training_samples/img_1.jpg \
    --name feature_fusion/concat_3:0 \
    --out_dir ./
```