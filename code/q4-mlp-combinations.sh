#!/bin/sh
epoch=20
bs=1


for lr in 0.001 0.01 0.1
do
    for hs in 100 200
    do
        for do in 0.3 0.5
        do
            for af in 'relu' 'tanh'
            do
                for op in 'sgd' 'adam'
                do
                    for ls in 1 2 3
                    do
                        filename='../images/4-MLP/results'"-LR"$lr"-HS"$hs"-DP"$do"-AF"$af"-OP"$op"-Ls"$ls'.txt'
                        res=$(python hw1-q4.py mlp -epochs $epoch -batch_size $bs -learning_rate $lr -hidden_sizes $hs -dropout $do -activation $af -optimizer $op -layers $ls)
                        $res >> $filename
                    done
                done
            done
        done
    done
done