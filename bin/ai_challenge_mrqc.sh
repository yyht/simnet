python trainer.py --model drcn \
 --model_config /notebooks/source/simnet/model_config.json \
 --model_dir /data/xuht/ai_challenge_cqmrc/nli/20181022/model/simnet \
 --config_prefix /notebooks/source/simnet/configs \
 --gpu_id 1 \
 --train_path "/data/xuht/ai_challenge_cqmrc/nli/20181022/train.txt" \
 --dev_path "/data/xuht/ai_challenge_cqmrc/nli/20181022/dev.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/ai_challenge_cqmrc/nli/20181022/yes_or_no/emb_mat_drcn.pkl" \
 --data_type "oqmrc" 
 # --elmo_w2v_path "/data/xuht/duplicate_sentence/LCQMC/elmo/elmo.pkl" \
 # --user_dict_path "/data/xuht/eventy_detection/inference_data/project_entity.txt" \
 