


set PATH_TO_PROCESS=D:/projects/phd_utils/blender_environments/20_09_24_fisheye_person/generated/tests
call pipenv --venv
pipenv run python dataset_handler/post_process_generated_series_test.py --config_path %PATH_TO_PROCESS% --write_movie True --delete_annotation False