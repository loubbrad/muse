folders:
	mkdir -p data/processed
	mkdir -p models
	mkdir -p samples
	
init: folders
	pip install -r req.txt
	pip install -r req-dev.txt

get_data: folders
	gsutil cp gs://muse-model/data/fugue_2048_256.json data/train.json
	gsutil cp gs://muse-model/data/fugue_prompts_2048.json data/prompt.json

get_maskedlm: folders
	gsutil cp gs://muse-model/models/pretrain_maskedlm/maskedlm_train0.01246_val0.0167.ckpt models/params.ckpt