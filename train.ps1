$config = "configs/ppyoloe_plus_crn_s_10e_coco.yml"
$log_dir = "log_dir/gdgrid_detection"
$weights = "https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_s_80e_coco.pdparams"

python -m paddle.distributed.launch `
    --log_dir=$log_dir `
    --gpus 0 PaddleYOLO/tools/train.py `
    -c $config `
    --eval `
    --amp `
    -o pretrain_weights=$weights
