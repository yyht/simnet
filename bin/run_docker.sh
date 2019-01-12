sudo nvidia-docker run -it -d -p 8418:8888 --name tensorflow_1.9_xuht_simnet_py3 \
-e LANG=C.UTF-8 \
-v /home/xuht/source:/notebooks/source \
-v /data/xuht:/data/xuht \
-p 6352:6006 \
-p 7399:8080 \
-p 8199:8891 \
-p 8237:8011 \
-e PASSWORD=123456 \
tensorflow/tensorflow:1.9.0-gpu-py3

sudo docker exec -it tensorflow_1.9_xuht_simnet_py3 bash
