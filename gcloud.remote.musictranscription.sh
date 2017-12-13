gcloud ml-engine jobs submit training "music_transcription_testH_$(date +%Y%m%d_%H%M%S)" \
	--job-dir gs://music_transciption_output \
	--module-name trainer.transcription \
	--package-path ./trainer \
	--region us-central1 \
	-- --X-file gs://music_transciption_output/X_input_full.npy --Y-file gs://music_transciption_output/Y_input_full.npy --id testH --job-dir gs://music_transciption_output/output --dropout-rate 0.5