init:
	pip install -r req.txt
	pip install -r req-dev.txt

data:
	mkdir -p data/processed

get_data: data
	wget https://www.dropbox.com/s/hjxo82decadlu3a/cpoint_chorales.json -P data/processed
	wget https://www.dropbox.com/s/lhl329l3149rjng/cpoint_fugues.json -P data/processed
	wget https://www.dropbox.com/s/gu63ls442eoasan/mutopia.json -P data/processed
	wget https://www.dropbox.com/s/l4zlynr4uo8hulf/kunstderfuge.json -P data/processed
	wget https://www.dropbox.com/s/rl4hlyfm0l4ros9/combined.json -P data/processed
	
train: init
	python muse/train.py
