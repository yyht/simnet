python test.py --model drcn \
 --model_config /notebooks/source/simnet/model_config.json \
 --model_dir /data/xuht/test/simnet \
 --config_prefix /notebooks/source/simnet/configs \
 --gpu_id 0 \
 --test_path "/data/xuht/duplicate_sentence/LCQMC/test.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/duplicate_sentence/LCQMC/emb_mat.pkl" \
 --model_str "drcn_1536928694_0.07039919369649823_0.8031129286631051" \
 #--vocab_path "/data/xuht/duplicate_sentence/LCQMC/elmo/emb_mat.pkl" \
 #--elmo_w2v_path "/data/xuht/duplicate_sentence/LCQMC/elmo/elmo.pkl"

