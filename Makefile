init:
	pip install -r req.txt
	pip install -r req-dev.txt

folders:
	mkdir -p data/processed
	mkdir -p models

get_data: folders
	wget https://www.dropbox.com/s/9p97xlcc4n8dx8i/mutopia_2048_128.json -P data/processed

get_params: folders
	wget https://www.dropbox.com/s/8j8shvjsdaveu0z/epoch%3D2-train_loss%3D0.027042848989367485-val_loss%3D0.02502560056746006.ckpt -P models
	
train: init
	python muse/train.py -m maskedlm-pretrain -c models/epoch=2-train_loss=0.027042848989367485-val_loss=0.02502560056746006.ckpt --workers 16 --gpus 8 --epochs 100
