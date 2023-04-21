folders:
	mkdir -p data
	mkdir -p models
	mkdir -p samples
	
init: folders
	pip install -r req.txt
	pip install -r req-dev.txt

fugue_data: folders
	gsutil cp gs://muse-model/data/fugue_2048_256.json data/train.json
	gsutil cp gs://muse-model/data/fugue_prompts_2048.json data/prompts.json

maskedlm_params: folders
	gsutil cp gs://muse-model/models/pretrain_maskedlm/maskedlm_train0.01246_val0.0167.ckpt models/params.ckpt

fugue_params: folders
	gsutil cp gs://muse-model/models/finetune_fugue/epoch=18-train_loss=0.1624893844127655-val_loss=0.154626727104187.ckpt models/params.ckpt