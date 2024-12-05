T5Chem
======

.. image:: https://img.shields.io/pypi/v/t5chem.svg
    :target: https://pypi.python.org/pypi/t5chem
    :alt: Latest PyPI version

A Unified Deep Learning Model for Multi-task Reaction Predictions.

It is built on `huggingface transformers`_ -- T5 model with some modifications.

.. image:: cover.png

.. _huggingface transformers: https://github.com/huggingface/transformers

Docker
------
We have a docker image available `here <https://hub.docker.com/repository/docker/hellojocelynlu/t5chem>`__, feel free to try it out!


Installation
------------

T5Chem can be either installed via pip or from source. We recommend to install t5chem from source.

1. To install from source

 .. code:: bash

   $ git clone https://github.com/HelloJocelynLu/t5chem.git
   $ cd t5chem/
   $ pip install .

It should automatically handle dependencies for you.

2. To install via pip

 .. code:: bash

   $ pip install t5chem

Usage
-----
Call from command line:

.. code:: bash

   $ t5chem -h # show the general help information
   $ t5chem train -h # show help information for model training
   $ t5chem predict -h # show help information for model prediction

We have some sample data (a small subset from datasets used in paper) available in `data/` folder, to have a quick start:

.. code:: bash

   $ tar -xjvf data/sample_data.tar.bz2
   $ t5chem train --data_dir data/sample/product/ --output_dir model/ --task_type product --num_epoch 30        # Train a model
   $ t5chem predict --data_dir data/sample/product/ --model_dir model/      # test a trained model

However, finetune based on a pre-trained model is strongly recommended. (You will get ~70% top-1 accuracy if training from a pretrained model by using `--pretrain`. But only 0.1% top-1 accuracy by training from scratch). You can download some pre-trained models and more datasets `here <https://zenodo.org/records/14280768>`__.

Download and extract the pretrained model weights:

.. code:: bash

   $ wget https://zenodo.org/records/14280768/files/simple_pretrain.tar.bz2
   $ tar -xjvf simple_pretrain.tar.bz2

Train using a pretrained model and test it:

.. code:: bash

   $ t5chem train --data_dir data/sample/product/ --output_dir model/ --task_type product --num_epoch 30 --pretrain models/pretrain/simple/
   $ t5chem predict --data_dir data/sample/product/ --model_dir model/

A more detailed example training from pretrained weights and explanations for commonly used arguments can be find `here <https://yzhang.hpc.nyu.edu/T5Chem/tutorial.html>`__.

Call as an API (Test a trained model):

.. code:: python

   from transformers import T5ForConditionalGeneration
   from t5chem import T5ForProperty, SimpleTokenizer
   pretrain_path = "path/to/your/pretrained/model/"
   model = T5ForConditionalGeneration.from_pretrained(pretrain_path)    # for seq2seq tasks
   tokenizer = SimpleTokenizer(vocab_file=os.path.join(pretrain_path, 'vocab.txt'))
   inputs = tokenizer.encode("Product:COC(=O)c1cc(COc2ccc(-c3ccccc3OC)cc2)c(C)o1.C1CCOC1>>", return_tensors='pt')
   output = model.generate(input_ids=inputs, max_length=300, early_stopping=True)
   tokenizer.decode(output[0], skip_special_tokens=True) # "COc1ccccc1-c1ccc(OCc2cc(C(=O)O)oc2C)cc1"

   model = T5ForProperty.from_pretrained(pretrain_path)  # for non-seq2seq task
   inputs = tokenizer.encode("Classification:COC(=O)c1cccc(C(=O)OC)c1>CN(C)N.Cl.O>COC(=O)c1cccc(C(=O)O)c1", return_tensors='pt')
   outputs = model(inputs)
   print(outputs.logits.argmax())   # Class 3

We have Google Colab examples available! Feel free to try it out:

- Call T5Chem via CLI (command line) `Colab <https://colab.research.google.com/drive/13tJlJ5loLtws6u91shbSjuPoiA1fCSae?usp=sharing>`__

- Use a pretrained model in python script `Colab <https://colab.research.google.com/drive/1xwz7c7q1SwwD5jEQKamo9TNCN1PKH8um?usp=sharing>`__

- Design your own project: predict molecular weights `Colab <https://colab.research.google.com/drive/1eu22gjGJDwXy59TBL8pfDmBF5_DQXBGn?usp=sharing>`__


Licence
-------
MIT Licence.

Authors
-------

`t5chem` was written by `Jocelyn Lu <jl8570@nyu.edu>`_.

Reference
----------

Jieyu Lu and Yingkai Zhang., Unified Deep Learning Model for Multitask Reaction Predictions with Explanation. *J. Chem. Inf. Model.*, **62**. 1376–1387 (2022) https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c01467

.. code:: bash

      @article{lu2022unified,
      title={Unified Deep Learning Model for Multitask Reaction Predictions with Explanation},
      author={Lu, Jieyu and Zhang, Yingkai},
      journal={Journal of Chemical Information and Modeling},
      year={2022},
      publisher={ACS Publications}
      }

Other projects in Zhang's Lab:
https://www.nyu.edu/projects/yzhang/IMA/
