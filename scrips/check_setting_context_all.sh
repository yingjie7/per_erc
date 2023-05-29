
#!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N etaskCT
#PBS -j oe  
#PBS -q GPU-1 
#PBS -o run.log
#PBS -e ierr.log

cd $PBS_O_WORKDIR

source ~/.bashrc
conda activate env_py38/  

python src/main.py --intra_speaker_context \
--dropout  0.2 --batch_size 1 --lr 1e-5 --max_ep 30 --froze_bert_layer 10 \
--data_name_pattern "iemocap.{}window2.json" --pre_trained_model_name roberta-large

python src/main.py --intra_speaker_context \
--dropout  0.2 --batch_size 1 --lr 5e-6 --max_ep 30 --froze_bert_layer 10 \
--data_name_pattern "meld.{}window2.json" --pre_trained_model_name roberta-large

python src/main.py --intra_speaker_context \
--dropout  0.2 --batch_size 1 --lr 5e-6 --max_ep 30 --froze_bert_layer 10 \
--data_name_pattern "emorynlp.{}window2.json" --pre_trained_model_name roberta-large

python src/main.py --intra_speaker_context  \
--dropout  0.2 --batch_size 1 --lr 5e-6 --max_ep 30 --froze_bert_layer 10 \
--data_name_pattern "dailydialog.{}window2.json" --pre_trained_model_name roberta-large

wait


echo "All done"