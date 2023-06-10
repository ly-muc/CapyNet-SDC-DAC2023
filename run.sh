DEPLOY_CFG_PATH=../mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py
MODEL_CFG_PATH=configs/runs/nanodet.py
MODEL_CHECKPOINT_PATH=nanodet.pth
INPUT_IMG=images/00002.jpg

python ./../mmdeploy/tools/deploy.py \
	${DEPLOY_CFG_PATH} \
	${MODEL_CFG_PATH} \
	${MODEL_CHECKPOINT_PATH} \
	${INPUT_IMG} \
	--work-dir work_dir \
	--device cuda:0 \
	--dump-info 
	 
	
