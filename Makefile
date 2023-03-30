init:
	pip install -r req.txt
	pip install -r req-dev.txt

data:
	mkdir -p data/processed

get_data: data
	wget https://www.dropbox.com/s/a82yxmtwypl9aoc/chorale_dataset.json -P data/processed
	wget https://www.dropbox.com/s/mmd0xrs4lzywyxg/mutopia.json -P data/processed
	
train: init
	python muse/train.py
