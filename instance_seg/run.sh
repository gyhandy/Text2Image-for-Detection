syn=$1
resnets=$2
lrs=$3
if [[ -d $syn ]]; then
   echo "$syn exists"
else
    exit -1;
fi;

for resnet in $(echo $resnets | tr "," "\n"); do
    for lr in $(echo $lrs | tr "," "\n"); do
        echo "$resnet with $lr"
        /lab/andy/anaconda3/envs/paste-segment/bin/python seg.py \
            -s syn -t voc_val \
            --blending gaussian \
            --lr $lr \
            --freeze --data_aug --crop \
            --epoch 20 \
            --resnet $resnet \
            --syn_dir $syn;
    done;
done;