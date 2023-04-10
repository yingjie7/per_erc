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

```cmd
    conda create --prefix=./env_py38  python=3.8
    conda activate ./env_py38 
    pip install -r requirements.txt
```
