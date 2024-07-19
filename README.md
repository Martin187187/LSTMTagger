# LSTM_Tagger

# Init

Commands I used to install requirements on my windows machine. I used python 3.12.
```command
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
```
Ensure that your directory structure looks like the following: 
(You have to have conll Data and the glove embedding in order to run this project.)
root/  
|-- data/  
|   |-- dev.conll  
|   |-- test.conll  
|   |-- train.conll  
|-- models/  
|-- utils/  
|-- glove.6B.50d.txt
|-- sequence_tagger.py  
|-- README.md
## Run

```command
python -u sequence_tagger.py --seed 420 --batch_size 1 --dropout_rate 0.2 --learning_rate 0.001
```
# Results

A plot will be generated with the performance of the train and dev set. 
In the end the test set will be evaluated on the model, which performed best on the dev set.


# Special Notes
- Data Handling: The system reads CoNLL data sentence by sentence and converts it to lowercase.
- Dropout Observation: After testing various dropout rates, it was observed that dropout 0 to 0.2 performed best.
- Dataset Performance: Even with higher dropout rates, the development dataset did not improve as much as the training dataset.
- Teacher Forcing: Experimented with teacher forcing but did not observe notable changes in performance.
- Learning Rate Scheduler: Implemented a scheduler to adjust the learning rate during training, but its impact was minimal.
- Batch Processing: Used collate_fn to ensure uniform batch length, although batch size is constrained to 1.
- Embedding Strategy: Used word embeddings as per requirements, although there was consideration for additional character embeddings
