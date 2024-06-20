
## Build and train your model locally in a Vertex Notebook
Note: this lab adapts and extends the official [TensorFlow BERT text classification tutorial](https://www.tensorflow.org/text/tutorials/classify_text_with_bert#define_your_model) to utilize Vertex AI services. See the tutorial for additional coverage on fine-tuning BERT models using TensorFlow.


[**Bidirectional Encoder Representations from Transformers (BERT)**](https://arxiv.org/abs/1810.04805v2) is a transformer-based text representation model pre-trained on massive datasets (3+ billion words) that can be fine-tuned for state-of-the art results on many natural language processing (NLP) tasks. Since release in 2018 by Google researchers, its has transformed the field of NLP research and come to form a core part of significant improvements to [Google Search](https://www.blog.google/products/search/search-language-understanding-bert). 

To meet your business requirements of achieving higher accuracy on a small dataset (20k training examples), you will use a technique called transfer learning to combine a pre-trained BERT encoder and classification layers to fine tune a new higher performing model for binary sentiment classification.

For this lab, you will use a smaller BERT model that trades some accuracy for faster training times.

The Small BERT models are instances of the original BERT architecture with a smaller number L of layers (i.e., residual blocks) combined with a smaller hidden size H and a matching smaller number A of attention heads, as published by

Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova: ["Well-Read Students Learn Better: On the Importance of Pre-training Compact Models"](https://arxiv.org/abs/1908.08962), 2019.

They have the same general architecture but fewer and/or smaller Transformer blocks, which lets you explore tradeoffs between speed, size and quality.

The following preprocessing and encoder models in the TensorFlow 2 SavedModel format use the implementation of BERT from the [TensorFlow Models Github repository](https://github.com/tensorflow/models/tree/master/official/nlp/bert) with the trained weights released by the authors of Small BERT.

Text inputs need to be transformed to numeric token ids and arranged in several Tensors before being input to BERT. TensorFlow Hub provides a matching preprocessing model for each of the BERT models discussed above, which implements this transformation using TF ops from the TF.text library. Since this text preprocessor is a TensorFlow model, It can be included in your model directly.

For fine-tuning, you will use the same optimizer that BERT was originally trained with: the "Adaptive Moments" (Adam). This optimizer minimizes the prediction loss and does regularization by weight decay (not using moments), which is also known as [AdamW](https://arxiv.org/abs/1711.05101).

For the learning rate `initial-learning-rate`, you will use the same schedule as BERT pre-training: linear decay of a notional initial learning rate, prefixed with a linear warm-up phase over the first 10% of training steps `n_warmup_steps`. In line with the BERT paper, the initial learning rate is smaller for fine-tuning.


### Build and compile a TensorFlow BERT sentiment classifier

Next, you will define and compile your model by assembling pre-built TF-Hub components and tf.keras layers.
**Note:** For any help while defining the model, you can refer [**TensorFlow BERT text classification tutorial**](https://www.tensorflow.org/text/tutorials/classify_text_with_bert#define_your_model).