# Instructions to run the project

1. Setup environment and install dependencies. Make sure that you have [poetry](https://python-poetry.org/docs/) and [pyenv](https://github.com/pyenv/pyenv).
    ```bash
    $ pyenv install 3.9.16
    $ pyenv virtualenv 3.9.16 summoner-spell-tracker
    $ pyenv activate summoner-spell-tracker
    $ cd summoner-spell-tracker
    $ poetry install
    ```
2. Extract frames from screenrecorded video:
    ```bash
    $ poetry run python src/create_dataset_from_video.py --process-video
    ```
3. Generate the dataset:
    ```bash
    $ poetry run python src/generate_data.py
    ```
3. (Optional) Check the generated dataset to view its statics or some sample images:
    ```bash
    $ poetry run python src/prepare_dataset.py
    ```
4. Train the model:
    ```bash
    $ poetry run python src/train.py --accelerator 'gpu' --max_epochs 15 --devices 1
    ```
5. All the results will be saved to the `tensorboard_logs` folder and all the generated data will be saved to the `data` folder.