SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_3d-esgs"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers

for SCENE in $SCENE_LIST;
do
    if false; then
        # Default.
        if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
            DATA_FACTOR=2
        else
            DATA_FACTOR=4
        fi
    else
        # Configuration on SGS.
        DATA_FACTOR=8
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python3 simple_trainer_3d-esgs.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir ../$SCENE_DIR/$SCENE/ \
        --result_dir ../$RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in ../$RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python3 simple_trainer_3d-esgs.py default --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir ../$SCENE_DIR/$SCENE/ \
            --result_dir ../$RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done

    echo "=== Eval Stats ==="

    for STATS in ../$RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in ../$RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done
