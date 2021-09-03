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
# python main.py +exp_v2=004_effnetb0_bandpass_fp16_cos #DEBUG=True
# python main.py +exp_v2=005_effnetb0_bandpass_fp16_cos_5.yaml



# python main.py +exp_v2=006_effnetb0_bandpass_fp32_cos_5.yaml
# python main.py +exp_v2=007_effnetb0_bandpass_fp32_cos_5_radam
# python main.py +exp_v2=008_rexnet150_bandpass_fp32_cos_5 
# python main.py +exp_v2=009_rexnet150_bandpass_fp32_cos_5_radam 
# python main.py +exp_v2=010_wavenet_bandpass_fp32_cos_10
# python main.py +exp_v2=011_rexnet150_bandpass_fp16_cos_5_radam_512

# python create_spec.py
# rm -rf input/img_fp16_256
# mkdir input/img_fp16_256
# python main.py +exp_v2=012_effnetb0_gwpy EXP_NAME=012_effnetb0_gwpy_wo_whiten_20_1024

# python main.py +exp_v2=014_swin_fp32_cos_5 #DEBUG=True
# python main.py -m +exp_v2=015_effnetb0_overfitted_OOF FOLD='range(0, 5)'

python main.py -m +exp_v2=016_effnetb0_sampled_batch.yaml