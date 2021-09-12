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

# python main.py -m +exp_v2=016_effnetb0_sampled_batch.yaml
# python main.py -m +exp_v2=017_wavenet_bandpass_fp32_cos_15_64bs
# python main.py +exp_v2=018_wavenet_bandpass_fp32_cos_15_400bs
# python main.py +exp_v2=019_effnetb0_fp32_big_spec
# python main.py +exp_v2=020_effnetb0_fp32_big_spec_30-500
# python main.py +exp_v2=021_effnetb0_fp32_bp30_800 DEBUG=True


# python main.py +exp_v2=022_effnetb0_standart_scaler_1ch
# python main.py +exp_v2=023_effnetb0_standart_scaler_3ch
# python main.py +exp_v2=024_effnetb0_minmax_scaler_1ch
# python main.py +exp_v2=025_effnetb0_minmax_scaler_3ch

# python main.py +exp_v2=026_effnetb0_standart_scaler_1ch_wo_minmax

# python main.py -m +exp_v2=028_wavenet_bandpass_fp32_cos_10_best.yaml FOLD='range(1, 5)'

# python main.py +exp_v2=027_effnetb0_bandpass_fp32_cos_5_best
# python main.py +exp_v2=029_effnetb0_bandpass_fp32_cos_15 # DEBUG=True
# python main.py +exp_v2=030_effnetb0_bandpass_fp32_cos_15_clip0.5
# python main.py +exp_v2=031_effnetb0_bandpass_fp32_cos_restarts_15_clip0.5
# python main.py +exp_v2=032_effnetb0_bandpass_fp32_timmcos_clip0.5.yaml
# python main.py +exp_v2=032_effnetb0_bandpass_fp32_timmcos_clip0.5.yaml BS=64 FP16=True EXP_NAME=031_effnetb0_bandpass_fp32_cos_restarts_15_clip0.5_bs64
# python main.py +exp_v2=033_wavenet_bandpass_fp16_cos_10_best_clip FOLD=3
# python main.py +exp_v2=033_wavenet_bandpass_fp16_cos_10_best_clip FOLD=1
# python main.py +exp_v2=033_wavenet_bandpass_fp16_cos_10_best_clip FOLD=4

# python main.py +exp_grid=baseline
# python main.py +exp_grid=baseline EXP_NAME=1 MODEL.CFG.fmin=30 MODEL.CFG.fmax=400
# python main.py +exp_grid=baseline EXP_NAME=2 TRANSFORM.CFG.lf=20 TRANSFORM.CFG.hf=500
# python main.py +exp_grid=baseline EXP_NAME=3 TRANSFORM.CFG.lf=20 TRANSFORM.CFG.hf=500 MODEL.CFG.fmin=20 MODEL.CFG.fmax=500
# python main.py +exp_grid=baseline EXP_NAME=4 TRANSFORM.CFG.lf=20 TRANSFORM.CFG.hf=600
# python main.py +exp_grid=baseline EXP_NAME=5 TRANSFORM.CFG.lf=40 TRANSFORM.CFG.hf=600
# python main.py +exp_grid=baseline EXP_NAME=6 TRANSFORM.CFG.lf=20 TRANSFORM.CFG.hf=700



# python main.py +exp_grid=baseline EXP_NAME=7 MODEL.CFG.hop_length=24
# python main.py +exp_grid=baseline EXP_NAME=8 MODEL.CFG.hop_length=16
# python main.py +exp_grid=baseline EXP_NAME=9 MODEL.CFG.hop_length=8

# python main.py +exp_grid=baseline EXP_NAME=10 MODEL.CFG.bins_per_octave=12
# python main.py +exp_grid=baseline EXP_NAME=11 MODEL.CFG.bins_per_octave=16
# python main.py +exp_grid=baseline EXP_NAME=12 MODEL.CFG.bins_per_octave=24

# python main.py +exp_grid=baseline EXP_NAME=13 MODEL.CFG.filter_scale=0.5
# python main.py +exp_grid=baseline EXP_NAME=14 MODEL.CFG.filter_scale=1.5
# python main.py +exp_grid=baseline EXP_NAME=15 MODEL.CFG.filter_scale=2
# python main.py +exp_grid=baseline EXP_NAME=16 MODEL.CFG.filter_scale=3

# python main.py +exp_grid=baseline EXP_NAME=17 MODEL.CFG.hop_length=24 MODEL.CFG.bins_per_octave=12 MODEL.CFG.filter_scale=0.5
# python main.py +exp_grid=baseline EXP_NAME=18 MODEL.CFG.hop_length=24 MODEL.CFG.bins_per_octave=12 MODEL.CFG.filter_scale=1.5
# python main.py +exp_grid=baseline EXP_NAME=19 MODEL.CFG.hop_length=16 MODEL.CFG.bins_per_octave=16 MODEL.CFG.filter_scale=1

# python main.py +exp_grid=baseline EXP_NAME=20 MODEL.CFG.hop_length=16 MODEL.CFG.bins_per_octave=16 MODEL.CFG.filter_scale=1 \
# TRANSFORM.CFG.lf=20 TRANSFORM.CFG.hf=500 MODEL.CFG.fmin=20 MODEL.CFG.fmax=500



# python main.py +exp_v2=034_wavenetv2_bandpass_fp32_cos_10_best MODEL.NAME=Wavenet_v3 DEBUG=True

# python main.py +exp_grid=baseline_v2 EXP_NAME=21
# python main.py +exp_grid=baseline_v2 EXP_NAME=22 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.bandpass_transform
# python main.py +exp_grid=baseline_v2 EXP_NAME=23 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.turkey_bandpass_transform
# python main.py +exp_grid=baseline_v2 EXP_NAME=24 TRANSFORM.NAME=src.transforms.minmax_turkey_bandpass_transform

# python main.py +exp_v2=034_wavenetv2_bandpass_fp32_cos_10_best MODEL.NAME=Wavenet_v3

# python main.py +exp_grid=baseline_v2 EXP_NAME=25 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.bandpass_transform \
# TRANSFORM.CFG.lf=20 TRANSFORM.CFG.hf=500 MODEL.CFG.fmin=20 MODEL.CFG.fmax=500 MODEL.CFG.bins_per_octave=12 MODEL.CFG.filter_scale=0.5





# python main.py +exp_grid=baseline_v2 EXP_NAME=26 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.bandpass_transform BS=64
# python main.py +exp_grid=baseline_v2 EXP_NAME=27 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.bandpass_transform BS=32
# python main.py +exp_grid=baseline_v2 EXP_NAME=28 MODEL.USE_SCALER=False TRANSFORM.NAME=src.transforms.minmax_turkey_bandpass_transform BS=64
# python main.py +exp_grid=baseline_v2 EXP_NAME=29 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.bandpass_transform BS=64 FP16=True DEBUG=True

# python main.py +exp_grid=baseline_v2 EXP_NAME=30 MODEL.USE_SCALER=False TRANSFORM.NAME=src.transforms.minmax_bandpass_transform BS=64

# python main.py +exp_grid=baseline_v2 EXP_NAME=26 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.turkey_bandpass_transform BS=64

# python main.py +exp_grid=baseline_b3_384 FOLD=0 
# python main.py +exp_grid=b0_512_v1 DEBUG=True
# python main.py +exp_grid=baseline_v2_cwt
# python main.py +exp_grid=baseline_v2_cwt_bandpass



# python main.py +exp_best=00_b0_128
# python main.py +exp_best=00_b0_128 EXP_NAME=00_b0_128_v2 MODEL.CFG.fmin=20 MODEL.CFG.fmax=1024
# python main.py +exp_best=00_b0_128 EXP_NAME=00_b0_128_v3 MODEL.CFG.fmin=20 MODEL.CFG.fmax=1024 MODEL.USE_SCALER=True
# python main.py +exp_best=00_b0_128 EXP_NAME=00_b0_128_v4 MODEL.CFG.fmin=20 MODEL.CFG.fmax=1024 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.minmax_bandpass_transform
# python main.py +exp_best=00_b0_128 EXP_NAME=00_b0_128_v5 MODEL.CFG.fmin=20 MODEL.CFG.fmax=1024 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.bandpass_transform
# python main.py +exp_best=00_b0_128 EXP_NAME=00_b0_128_v6 MODEL.USE_SCALER=True TRANSFORM.NAME=src.transforms.bandpass_transform


# python main.py +exp_grid=baseline_v2_cwt_bandpass_scaler
# python main.py +exp_v2=033_wavenet_bandpass_fp16_cos_10_best_clip FOLD=0 BS=64 EXP_NAME=wavenet_bs64_clip

# python main.py +exp_best=00_b0_128_best EXP_NAME=00_b0_128_best BS=32 FOLD=1

# python main.py +exp_best=01_b3_512 DEBUG=True

# python main.py +wavenets=001
# python main.py +wavenets=000_cnn_baseline128 BS=32
# python main.py +wavenets=002
# python main.py +wavenets=003
# python main.py +wavenets=004


# python main.py +exp_multi_cqt=00_b0_128_5cqt # DEBUG=True
# python main.py -m +exp_multi_cqt=01_b0_512_5cqt FOLD='range(0, 5)'
# python main.py +exp_multi_cqt=01_b0_512_5cqt FOLD=2 MODEL.CHECKPOINT=weights/CustomModel_v2/00_b0_512_5cqt/fold_2/cp_epoch00_score0.87026.pth

# python create_submission.py +exp_multi_cqt=01_b0_512_5cqt

# python main.py +exp_multi_cqt=02_b4_512_5cqt FOLD=0 MODEL.CHECKPOINT=weights/CustomModel_v2/02_b4_512_5cqt/fold_0/cp_epoch00_score0.87082.pth #causes nan at val

# python main.py +exp_multi_cqt=03_rexnet_512_5cqt.yaml FOLD=0 DEBUG=True
python main.py +exp_multi_cqt=06_512x512_b4_f16_scaler
python main.py +exp_multi_cqt=05_256x512_b4_f32 FOLD=0
python main.py +exp_multi_cqt=05_256x512_b4_f32 FOLD=1
python main.py +exp_multi_cqt=05_256x512_b4_f32 FOLD=2
python main.py +exp_multi_cqt=05_256x512_b4_f32 FOLD=3
