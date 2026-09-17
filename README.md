# T-FARMA

This is the code of T-FARMA.

## How to run

#### Process data

For all the datasets 'ICEWS18', 'ICEWS14', 'ICEWS05-15', 'GDELT' ,  'WIKI', the following command can be used to generate the historical frequency matrix.
```shell
cd src
python get_history.py --dataset ICEWS14
```

#### Train models

Then the following commands can be used to train T-FARMA.

Train models

```shell
cd src
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 14 --discount 1  --gpu 0 --n-epochs 15 --use-cl --rel-weight 0.1
```



#### Evaluate models

The following commands can be used to evaluate T-FARMA.

```shell
cd src
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --angle 14 --discount 1 --gpu 0 --n-epochs 15 --use-cl --rel-weight 0.1 -tfa-weight 0.2 --test
```



