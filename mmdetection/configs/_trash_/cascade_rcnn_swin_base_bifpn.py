_base_ = [
    './_base_/datasets/coco_detection.py',
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth'  # noqa
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 128*2, 128*4, 128*8]))

neck=dict(
    type='BiFPN',
    in_channels=[128, 128*2, 128*4, 128*8],
    out_channels=128,
    num_outs=5)