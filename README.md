# audio2viseme
The code generate viseme from audio features.

## Installation

#### The project depends on the following Python packages:

* Python--- 3.8.2
* Keras --- 2.4.1
* Tensorflow --- 2.4.1
* Librosa --- 0.8.0
* matplotlib
* numpy
* tqdm


## Audio Feature Extraction

You can run audio_feature_extractor.py to extract audio features from audio files. The arguments are as follows:

* -i --- Input folder containing audio files (if your audio file types are different from .wav, please modify the script accordingly)
* -d --- Delay in terms of frames, where one frame is 40 ms 
* -c --- Number of context frames
* -o --- Output folder path to save audio feature 

Usage:

```
python audio_feature_extractor.py -i path-to-audio-files/ -d number-of-delay-frames -c number-of-contex-frames -o output-folder-to-save-audio-feature-file
```

## Training

The training code has the following arguments:

* -x --- Input audio feature file for training
* -X --- Input audio feature file for testing
* -y --- Input viseme file for training
* -Y --- Input viseme file for testing
* -o --- Output folder path to save the model

Usage:

```
python train.py -x path-to-training-audio-feature-file, -X path-to-testing-audio-feature-file, -y path-to-training-viseme-file -Y path-to-testing-viseme-file -o path-to-save-model-file

```

