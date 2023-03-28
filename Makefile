init:
	conda create -n muse python
	conda activate muse
	pip install -r req.txt
	mkdir -p data/processed
	wget https://www.dropbox.com/s/a82yxmtwypl9aoc/chorale_dataset.json -P data/processed
	wget https://www.dropbox.com/s/mmd0xrs4lzywyxg/mutopia.json -P data/processed
