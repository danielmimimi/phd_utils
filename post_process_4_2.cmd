


set PATH_TO_PROCESS=D:/gen4_2
call pipenv --venv
pipenv run python dataset_handler/post_process_generated_series_gen4_3.py --config_path %PATH_TO_PROCESS% --write_movie False