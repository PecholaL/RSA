# RSAno
Restorable Speaker Anonymization via Invertible Neural Network

# Training RSAno
## Data Processing
1. Modify `./data/config.yaml` according to the location of your prepared dataset.
2. Preprocess the .wav files, build Dataset and dump into pickle:
    `python3 ./data/preprocess.py`
3. Test your Dataset and DataLoader:
    `python3 ./data/test_data.py`

## Model Building
