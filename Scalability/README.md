# How to reproduce the experiments
## Dependencies

**Recommended: Python 3.7**

Please, ensure you use Python 3.7. Otherwise, we do not guarantee the correct installation of dependencies.

You can install all the dependencies by running the following commands.
```
pip install -r requirements.txt
```

## Download the data
You can download the datasets for this particular experiment here:
- [Scalability Dataset](https://bnn.upc.edu/download/dataset-v5-scalability/)

Otherwise, you can download the datasets using the following command:
```
wget -O scalability.zip https://bnn.upc.edu/download/dataset-v5-scalability/
```

Note that this dataset is a zip file, so you need to decompress it first. Also, these experiments suppose that the data 
is in a directory file called `data` located at the root of the repository. If you want to change this, you can change the
path sent as input to the `input_fn` function.

## Training and Validation
If you want to train the model, you can use the following command:
```
python main.py
```

If everything has been done correctly, you should see the following output:

```
Starting training from scratch...
Epoch 1/200
36/4000 [..............................] - ETA: 3:06 - loss: 156.5287 - denorm_MAPE: 191.0536
```

The model will train during 200 epochs. You can change this number by changing the `epochs` parameter in the `fit` function.

Finally, the models will be saved in a `ckp_dir` directory. You can change this path by changing the `ckp_dir` variable. 

## Prediction
We have already provided the trained models for each one of the experiments. If you want, you can skip the Training and Validation
part and just use the following command to predict the metrics:
```
python check_prediction.py
```
This script will, first, load the best model located in the `ckp_dir` directory. Then, it will evaluate the metrics for
each one of the sample datasets.

Again, if you configured everything correctly, you should see something like this:
```
BEST CHECKOINT FOUND: 69-9.91
PREDICTING...
Directory: ../data/scalability/test\50
2022-02-28 18:46:19.864938: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
COLLECTING...
1
MAPE: 7.31 %
2
MAPE: 5.98 %
```

## License
See [LICENSE](LICENSE) for full of the license text.

```
Copyright Copyright 2022 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
