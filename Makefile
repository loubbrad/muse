init:
	pip install -r req.txt
	pip install -r req-dev.txt

folders:
	mkdir -p data/processed
	mkdir -p models

get_data: folders
	wget https://www.dropbox.com/s/b22yu9a5g5pgi6m/mutopia_2048_128.json -P data/processed
	wget https://www.dropbox.com/s/no5xi3oxjtxec5n/chorale_2048_128.json -P data/processed
	wget https://www.dropbox.com/s/9sc6m9dk6bu0hz5/fugue_2048_128.json -P data/processed

get_maskedlm: folders
	wget -O params.ckpt https://www.dropbox.com/s/87qapkjqsqvosgo/maskedlm_train0.01246_val0.0167.ckpt -P models
	
get_casual: folders
	wget -O params.ckpt https://www.dropbox.com/s/w5raludplckwiai/casual_train0.4273_val0.5335.ckpt -P models
	
download: get_data get_params
	echo Downloading...