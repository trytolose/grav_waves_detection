# python main.py +exp=exp_1 EXP_NAME=testtest
# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_turkey_transform_cwt FOLD=1
# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_turkey_transform_cwt FOLD=2
# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_turkey_transform_cwt FOLD=3
# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_turkey_transform_cwt FOLD=4

# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_turkey_transform_q TRANSFORM=src.transforms.stack_bandpass_turkey_transform FOLD=2
# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_turkey_transform_q TRANSFORM=src.transforms.stack_bandpass_turkey_transform FOLD=3
# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_turkey_transform_q TRANSFORM=src.transforms.stack_bandpass_turkey_transform FOLD=4
# python main.py +exp=exp_0 EXP_NAME=stack_bandpass_transform_q TRANSFORM=src.transforms.stack_bandpass_transform
# python main.py +exp=exp_0 EXP_NAME=stack_turkey_q TRANSFORM=src.transforms.stack_turkey


# python main.py +exp=exp_1
# python main.py +exp=exp_2
# python main.py +exp=exp_3
# python main.py +exp=exp_4
# python main.py +exp=exp_5
# python main.py +exp=exp_6_vit FOLD=0 EXP_NAME=wavenet
# python main.py +exp=exp_6_vit FOLD=0 EXP_NAME=test_test
# python main.py +exp=exp_6_vit FOLD=1 EXP_NAME=wavenet_att
# python main.py +exp=exp_6_vit FOLD=2 EXP_NAME=wavenet_att
# python main.py +exp=exp_6_vit FOLD=3 EXP_NAME=wavenet_att
# python main.py +exp=exp_6_vit FOLD=4 EXP_NAME=wavenet_att
# python main.py +exp=exp_6_vit FOLD=1
# python main.py +exp=exp_6_vit FOLD=2
# python main.py +exp=exp_6_vit FOLD=3
# python main.py +exp=exp_6_vit FOLD=4

# python main.py +exp_v2=000_wavenet_wo_bandpass.yaml
# python main.py +exp_v2=001_wavenet_base_plus_bandpass # DEBUG=True
#python main.py +exp_v2=004_effnetb0_bandpass_fp16_cos #DEBUG=True
#python main.py +exp_b=effnetb0_baseline_improve #DEBUG=True
#python main.py +exp_b=effnetb0_substract_noise #DEBUG=True


# python main.py -m +exp_b=effnetb0_baseline_improve MODEL.CFG.scale_cqt=True BS=32 FOLD="range(0, 5)" EXP_NAME=001_effnetb0_bs32_scale_cqt
# python main.py +exp_b=effnetb0_baseline_improve BS=48 FOLD=0 EXP_NAME=001_effnetb0_bs48
# python main.py +exp_b=effnetb0_baseline_improve MODEL.CFG.scale_cqt=True BS=48 FOLD=0 EXP_NAME=001_effnetb0_bs48_scale_cqt


# python main.py +exp_b=effnetb0_focal_loss LOSS.CFG.alpha=0.2 LOSS.CFG.gamma=1 BS=64 EXP_NAME=001_effnetb0_focal_loss_0.2_1
# python main.py +exp_b=effnetb0_focal_loss LOSS.CFG.alpha=0.2 LOSS.CFG.gamma=0.5 BS=64 EXP_NAME=001_effnetb0_focal_loss_0.2_0.5
# python main.py +exp_b=effnetb0_focal_loss LOSS.CFG.alpha=0.1 LOSS.CFG.gamma=1 BS=64 EXP_NAME=001_effnetb0_focal_loss_0.1_1

python main.py +exp_b=effnetb5_Q3
python main.py +exp_b=effnetb7_Q3


#python main.py +exp_b=effnetb0_baseline_improve FOLD=0
#python main.py +exp_b=effnetb0_focal_loss FOLD=0
#python main.py +exp_b=effnetb0_baseline_improve BS=32 FOLD=0 &&
#python main.py +exp_b=effnetb0_baseline_improve BS=32 FOLD=1 &&
#python main.py +exp_b=effnetb0_baseline_improve BS=32 FOLD=2 &&
#python main.py +exp_b=effnetb0_baseline_improve BS=32 FOLD=3 &&
#python main.py +exp_b=effnetb0_baseline_improve BS=32 FOLD=4 
