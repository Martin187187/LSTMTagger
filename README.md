﻿# LSTM_Tagger

## Init
```command
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Run

```command
python sequence_tagger.py --seed 42 --batch_size 1 --dropout_rate 0.2 --learning_rate 0.001
```
# Results

I put a plot with my result from the command from above. Its has a macro averaged F1 of 0.8 for the Test dataset.


Upon running, a plot will be generated in the same directory.

Special Notes
- Data Handling: The system reads CoNLL data sentence by sentence and converts it to lowercase.
- Dropout Observation: After testing various dropout rates, it was observed that no dropout performed significantly better.
- Dataset Performance: Even with higher dropout rates, the development dataset did not improve as much as the training dataset.
- Teacher Forcing: Experimented with teacher forcing but did not observe notable changes in performance.
- Learning Rate Scheduler: Implemented a scheduler to adjust the learning rate during training, but its impact was minimal.
- Batch Processing: Used collate_fn to ensure uniform batch length, although batch size is constrained to 1.
- Embedding Strategy: Used word embeddings as per requirements, although there was consideration for additional character embeddings
