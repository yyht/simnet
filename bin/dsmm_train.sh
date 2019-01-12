python dsmm_trainer.py --model dsmm_esim \
 --model_config "/notebooks/source/simnet/model_config.json" \
 --model_dir "/data/xuht/nli/model" \
 --config_prefix "/notebooks/source/simnet/configs" \
 --gpu_id "2" \
 --train_path "/data/xuht/nli/train.txt" \
 --dev_path "/data/xuht/nli/dev.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/nli/model/dsmm_esim_bigru/emb_mat.pkl" \
 --data_type "wsdm_nli"
 # --elmo_w2v_path "/data/xuht/duplicate_sentence/LCQMC/elmo/elmo.pkl"
 # --user_dict_path "/data/xuht/eventy_detection/inference_data/project_entity.txt" \
