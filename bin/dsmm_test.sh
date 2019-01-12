python dsmm_test.py --model decatt \
 --model_config /notebooks/source/simnet/model_config.json \
 --model_dir /data/xuht/test/simnet \
 --config_prefix /notebooks/source/simnet/configs \
 --gpu_id 2 \
 --test_path "/data/xuht/duplicate_sentence/LCQMC/test.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/duplicate_sentence/LCQMC/dsmm/emb_mat.pkl" \
 --model_str "decatt_1539063998_0.05873851926088333_0.8285995500069658" \
 #--vocab_path "/data/xuht/duplicate_sentence/LCQMC/elmo/emb_mat.pkl" \
 #--elmo_w2v_path "/data/xuht/duplicate_sentence/LCQMC/elmo/elmo.pkl"

