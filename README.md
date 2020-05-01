# Machine-Translation-System
English-to-Hindi language translation model using neural machine translation with seq2seq architecture.

# Dependencies
### Tensorflow<br/>
### Numpy

# Dataset
Dataset used for training was taken from http://www.manythings.org/anki/.
File contains 2778 parallel english and hindi sentences.

# Model
Model uses a sequence to sequence architecture along with Attention. <br />
Encoder model is an **LSTM** along with embeddings. <br />
Decoder model is an **LSTM** with Attention mechanism. <br />
![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Feeding-Hidden-State-as-Input-to-Decoder.png)

Attention Mechanism used is a **Luong Attention** <br/>
![](https://blog.floydhub.com/content/images/2019/09/Slide51.JPG)
