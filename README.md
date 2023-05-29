## Personalize emotional recognition in Conversation
This project aim to construct a deep learning model to recognize emotional label of utterances in a conversation. 
##  Data  
- IEMOCAP
    Data structure examples: 
    ```json
    [
        # this is first conversation 
        { 
            "labels": [
            4,
            2,
            4,
            4 
            ],
            "sentences": [
            "Guess what?",
            "what?",
            "I did it, I asked her to marry me.",
            "Yes, I did it."
            ],
            "sentences_mixed_around": [
                ...
            ],
            "s_id": "Ses05M_impro03",
            "genders": [
            "M",
            "F",
            "M",
            "M",
            "F", 
            ]
        },

        # this is second conversation 
        { 
            "labels": [
            4,
            2,
            ],
            "sentences": [
            "Guess what?",
            "what?", 
            ],
            "sentences_mixed_around": [
                ...
            ],
            "s_id": "Ses05M_impro03",
            "genders": [
            "M",
            "F",  
            ]
        }
    ]
    ```

##  Python ENV 
Init python environment 
```cmd
    conda create --prefix=./env_py38  python=3.8
    conda activate ./env_py38 
    pip install -r requirements.txt
```

## Run 
1. Init environment follow the above step.
2. Data peprocessing. 
   1. Put all the raw data to the folder `data/`.
    The overview of data structure:
        ```
            .
            ├── data
            │   ├── meld.test.json
            │   ├── meld.train.json
            │   ├── meld.valid.json
            │   ├── iemocap.test.json
            │   ├── iemocap.train.json
            │   └── iemocap.valid.json
            └── ...
        ```
        From the root folder (curent folder), run data preprocess by `python data/raw_data_preprocessor.py`. The arround utterances of considering utterance (e.g., the local window is usually set to 2) is connected together for gathering the local context information. Some new files will be generated: 
        ```
            .
            ├── data
            │   ├──  ...
            │   ├── iemocap.testwindow2.json
            │   ├── iemocap.trainwindow2.json
            │   └── iemocap.validwindow2.json
            └── ...
        ```
3. Train  
    Run following command to train a new model. 
    ```bash 
    bash scrips/check_setting_context_all.sh
    ```
    > **Note**: Please check this scripts to check the setting and choose which data you want to run. 
