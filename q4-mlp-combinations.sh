#!/bin/sh
epoch=20
bs=1
lr=0.001
hs=100
do=0.3
af='relu'
op='sgd'
for ls in 1 2 3
do
    for lr in 0.1 0.01 0.001
    do
        python hw1-q4.py mlp -epochs $epoch -batch_size $bs -learning_rate $lr -hidden_sizes $hs -dropout $do -activation $af -optimizer $op -layers $ls
    done

    python hw1-q4.py mlp -epochs $epoch -batch_size $bs -learning_rate $lr -hidden_sizes 200 -dropout $do -activation $af -optimizer $op -layers $ls

    python hw1-q4.py mlp -epochs $epoch -batch_size $bs -learning_rate $lr -hidden_sizes $hs -dropout 0.5 -activation $af -optimizer $op -layers $ls
    
    python hw1-q4.py mlp -epochs $epoch -batch_size $bs -learning_rate $lr -hidden_sizes $hs -dropout $do -activation tanh -optimizer $op -layers $ls
    
    python hw1-q4.py mlp -epochs $epoch -batch_size $bs -learning_rate $lr -hidden_sizes $hs -dropout $do -activation $af -optimizer adam -layers $ls
done