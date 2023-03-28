init:
	git clone https://github.com/loua19/muse
    pip install -r requirements.txt
	cd muse
	mkdir -p data/processed
	wget https://www.dropbox.com/s/a82yxmtwypl9aoc/chorale_dataset.json -P data/processed
	wget https://www.dropbox.com/s/mmd0xrs4lzywyxg/mutopia.json -P data/processed

train: data
    python muse/train.py