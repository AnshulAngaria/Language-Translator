# Machine-Translation-System
Neural Machine Translator from English to Hindi

# Dependencies
Tensorflow<br/>
Numpy

# Dataset
Dataset used for training was taken from http://www.manythings.org/anki/.
File contains 2778 parallel english and hindi sentences.

# Model
Model uses a sequence to sequence architecture along with Attention. <br />
Encoder model is an LSTM along with embeddings. <br />
Decoder model is an LSTM with Attention mechanism. <br />
![](https://cdn-images-1.medium.com/max/1600/1*75Jb0q3sX1GDYmJSfl-gOw.gif)

<br/>Attention Mechanism used is a Luong Attention <br/>
![](https://blog.floydhub.com/content/images/2019/09/Slide51.JPG)
