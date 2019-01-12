python trainer.py --model bimpm \
 --model_config "/notebooks/source/simnet/model_config.json" \
 --model_dir "/data/xuht/nli/model" \
 --config_prefix "/notebooks/source/simnet/configs" \
 --gpu_id "1" \
 --train_path "/data/xuht/nli/train.txt" \
 --dev_path "/data/xuht/nli/dev.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/duplicate_sentence/LCQMC/man/emb_mat.pkl" \
 --elmo_w2v_path "/data/xuht/duplicate_sentence/LCQMC/elmo/elmo.pkl"

