python dsmm_trainer.py --model dsmm_esim \
 --model_config /notebooks/source/simnet/model_config.json \
 --model_dir /data/xuht/ai_challenge_cqmrc/nli/20181023/model/simnet \
 --config_prefix /notebooks/source/simnet/configs \
 --gpu_id 3 \
 --train_path "/data/xuht/ai_challenge_cqmrc/nli/20181023/train.txt" \
 --dev_path "/data/xuht/ai_challenge_cqmrc/nli/20181023/dev.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/ai_challenge_cqmrc/nli/20181023/vocab/emb_mat_dsmm_esim_gru.pkl" \
 --data_type "mrc" 
 # --user_dict_path "/data/xuht/eventy_detection/inference_data/project_entity.txt" 
 # --elmo_w2v_path "/data/xuht/duplicate_sentence/LCQMC/elmo/elmo.pkl"

