## step using LLM features 

1. extract LLM vector context of each utterance for each conversation (`llm_context_vect`)
2. modify the `DataLoader` process to load `llm_context_vect`
   considering bellow code: 
   1. given an conversation id `s_id`, and utternnce `i-th` load pickle file and pick the vector by bellowing example code:`
        ```python 
        import pickle 
        all_conversations = pickle.load(open('path/to/the/train.pkl'))
        s_id = "ses05_12ABXAAB"
        utterance_idx = 5
        llm_context_vect = all_conversations[s_id][utterance_idx]['llm_vect']
        ```
    2. change the dataloader process to load `llm_context_vector`
        ![img1](../images/load1.png)
    3. return context vector for each batch loader
        ![img1](../images/return1.png)
        
3. process llm_context_vector by concatinating with output vector  `u_vector_fused_by_context`
        ![img1](../images/process.png)
4. training a new model 