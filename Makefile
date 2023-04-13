init:
	pip install -r req.txt
	pip install -r req-dev.txt

data:
	mkdir -p data/processed

get_data: data
	wget https://www.dropbox.com/s/9p97xlcc4n8dx8i/mutopia_2048_128.json -P data/processed
	
train: init
	python muse/train.py -m maskedlm-pretrain --workers 16 --gpus 4 --epochs 100
