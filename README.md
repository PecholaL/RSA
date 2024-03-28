# RSA
[![](https://img.shields.io/badge/LICENSE-Apache_2.0-yellow?style=flat)](https://github.com/PecholaL/RSA/blob/main/LICENSE) 
[![](https://img.shields.io/badge/AI-speech-pink?style=flat)](https://github.com/PecholaL/RSA) 
[![](https://img.shields.io/badge/Pechola_L-blue?style=flat)](https://github.com/PecholaL)  

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
The training of RSA needs two pre-trained models: ACG and the speaker encoder. The pre-trained SpkEnc is taken from AutoVC. Modify the paths to .ckpt, .yaml, etc. in `./train_RSA.py`. Then excute:  
`python3 ./train_RSA.py`


## Inference
### Anonymization & Restoration
After the two-stage training of RSA, the anonymization and restoration processes can be conducted with a pre-trained vocoder (like [Hifi-GAN](https://github.com/jik876/hifi-gan)). Modify the *paths* in `./inference.py`, including the paths of the original .wav file, output path, pre-trained models' .ckpt. Then excute:  
`python3 ./inference.py`

## Thanks
[SpeechBrain](https://github.com/speechbrain/speechbrain)  
[FrEIA](https://github.com/vislearn/FrEIA)  
[cINN](https://github.com/vislearn/conditional_INNs)  
[Hifi-GAN](https://github.com/jik876/hifi-gan)  


## Citation
coming soon
