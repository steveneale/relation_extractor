# relation_extractor

*relation_extractor* is a program for training relation extraction models, written in Python and with models implemented using [TensorFlow](https://www.tensorflow.org). The model architecture is a bi-directional LSTM with attention, as outlined in the paper by Zhou et al. [[1](#paper1)].

The implementation here is in part an object-oriented reimagining of [this](https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction) version from [SeoSangwoo](https://github.com/SeoSangwoo), with some additional tweaks.

Models trained using the current implementation reach **77.28** F1 (macro-averaged [official score](https://github.com/steveneale/relation_extractor/blob/master/data/SemEval2010_task8_all_data/SemEval2010_task8_testing/README.txt) for SemEval2010 Task #8) - a little shy of the **84.00** reported in the paper, so still some improvements to be made.


## References

[<a name="paper1">1</a>] Zhou, P., Shi, W., Tian, J., Qi, Z., Li, B., Hao, H. and Xu, B. (2016) "Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification". *Proceedings of the 54th Annual Meeting of the Association for Computational Linguitsics (ACL '16)* [[pdf](http://www.aclweb.org/anthology/P16-2034)]