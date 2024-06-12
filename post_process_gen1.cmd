


set PATH_TO_PROCESS=D:/generated_annotated
call pipenv --venv
pipenv run python dataset_handler/post_process_generated_series.py --config_path %PATH_TO_PROCESS%