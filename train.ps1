$config = "configs/yolov5_s_100e_coco.yml"
$log_dir = "log_dir/gdgrid_detection"
$weights = "https://bj.bcebos.com/v1/paddledet/models/yolov5_s_300e_coco.pdparams"

python -m paddle.distributed.launch `
    --log_dir=$log_dir `
    --gpus 0 PaddleYOLO/tools/train.py `
    -c $config `
    --eval `
    --amp `
    -o pretrain_weights=$weights
