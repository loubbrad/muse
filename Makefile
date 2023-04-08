init:
	pip install -r req.txt
	pip install -r req-dev.txt

data:
	mkdir -p data/processed

get_data: data
	wget https://www.dropbox.com/s/hjxo82decadlu3a/cpoint_chorales.json -P data/processed
	wget https://www.dropbox.com/s/lhl329l3149rjng/cpoint_fugues.json -P data/processed
	
train: init
	python muse/train.py
