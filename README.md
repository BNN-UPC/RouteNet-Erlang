# RouteNet-Erlang

This is the official implementation of RouteNet-Erlang, a pioneering GNN architecture designed to model computer 
networks. RouteNet-Erlang supports complex traffic models, multi-queue scheduling policies, routing policies and 
can provide accurate estimates in networks not seen in the training phase.

## Quick Start
### Project structure

The project is divided into three main blocks: scheduling, traffic models, and scalability. Each block has its own 
directory and contains its own files. In each directory we can find four main files:
- `model.py`: contains the code of the GNN model.
- `main.py`: contains the code for the training/validation of the model.
- `config.ini`: contains the configuration of the experiment. Number of epochs, batch size, learning rate, number of
neurons, etc.
- `check_predictions.py`: contains the code for the predictions. It automatically loads the best-trained model and 
saves its predictions in a `predictions.npy` file that can be read using NumPy.

The project also contains some auxiliary files like `datanetAPI.py` used to read the different datasets and the 
`read_dataset.py` that is used to convert the samples provided by the dataset API into the graph that is taken as
input by the model.

You can find more information about the reproducibility of the experiments inside each one of the directories 
([Scheduling](/Scheduling/README.md), [Traffic Models](/TrafficModels/README.md), [Scalability](/Scalability/README.md)).

## Main Contributors
#### M. Ferriol-Galmés, J. Suárez-Varela, P. Barlet-Ros, A. Cabellos-Aparicio.

[Barcelona Neural Networking center](https://bnn.upc.edu/), Universitat Politècnica de Catalunya

#### Do you want to contribute to this open-source project? Please, read our guidelines on [How to contribute](CONTRIBUTING.md)

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