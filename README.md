# RSA
Restorable Speaker Anonymization via Invertible Neural Network

## Training
### Data Processing
1. Modify `./data/config.yaml` according to the location of your prepared dataset.
2. Preprocess the .wav files, build Dataset and dump into pickle:
    `python3 ./data/preprocess.py`
3. Test your Dataset and DataLoader:
    `python3 ./data/test_data.py`

### Model Building & Training
#### ASV
A pre-trained model from [SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) is utilized as the speaker embedding extractor for the training processes of ACG and RSA. The **EncoderClassifier** obtains 192-d speaker embedding from a piece of utterance. The package **pretrained_models** is built to link the packages of SpeechBrain, so modify the *BASE_DIR* in `./speech_brain_proxy/__init__.py` according to the path to the downloaded [*speechbrain-dev*](https://github.com/speechbrain/speechbrain).

#### ACG
The first training stage is for ACG. Use the preprocessed .pkl which contains longer audio segments to extract speaker embeddings. Modify the path to .pkl in `./train_ACG.py` and the training factors (e.g. learning rate, n_iterations, batch_size, etc) in `./models/config.yaml`. Then excute:  
`python3 ./train_ACG.py`

#### RSA


## Inference
### Anonymization

### Restoring

## Thanks
[SpeechBrain](https://github.com/speechbrain/speechbrain)
[FrEIA](https://github.com/vislearn/FrEIA)
[cINN](https://github.com/vislearn/conditional_INNs)


## Citation
