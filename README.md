# RouteNet-Erlang

This is the official implementation of RouteNet-Erlang, a pioneering GNN architecture designed to model computer 
networks. RouteNet-Erlang supports complex traffic models, multi-queue scheduling policies, routing policies and 
can provide accurate estimates in networks not seen in the training phase.

## Quick Start
### Dependencies

**Recommended: Python 3.7**

Please, ensure you use Python 3.7. Otherwise, we do not guarantee the correct installation of dependencies.

You can install all the dependencies by running the following commands.
```
pip install -r requirements.txt
```

### Datasets
You can download the datasets for each experiment here:
- [Scheduling Dataset]()
- [Traffic Models Dataset]()
- [Scalability Dataset]()

Otherwhise you can download the datasets using the following commands:
- Scheduling Dataset:
```
wget -O scheduling.zip http://www.domain.com/filename-4.0.1.zip
```
- Traffic Models Dataset:
```
wget -O traffic_models.zip http://www.domain.com/filename-4.0.1.zip
```
- Scalability Dataset:
```
wget -O scalability.zip http://www.domain.com/filename-4.0.1.zip
```

### Project structure

The project is divided into three main blocks: scheduling, traffic models and scalability. Each block has its own 
directory and contains its own files. In each directory we can find four main files:
- `model.py`: contains the code of the GNN model.
- `main.py`: contains the code code for the training/validation of the model.
- `config.ini`: contains the configuration of the experiment. Number of epochs, batch size, learning rate, number of
neurons, etc.
- `check_predictions.py`: contains the code for the predictions. It automatically loads the best trained model and 
saves its predictions in a `predicitons.npy` file taht can be read using numpy.

The project also contains some auxiliary files like `datanetAPI.py` used to read the different datasets and the 
`read_dataset.py` that is used to convert the samples provided by the dataset API into the graph that is taken as
input by the model.

## Main Contributors
#### M. Ferriol-Galmés, K. Rusek, J. Suárez-Varela, P. Barlet-Ros, A. Cabellos-Aparicio.

[Barcelona Neural Networking center](https://bnn.upc.edu/), Universitat Politècnica de Catalunya

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

##Community by-laws

RouteNet-Erlang is a pioneering GNN architecture designed to model computer networks. RouteNet-Erlang supports complex 
traffic models, multi-queue scheduling policies, routing policies and can provide accurate estimates in networks not 
seen in the training phase.

### Community Roles

Participants in the project can assume the following roles:

**User:** someone using the framework code, who contributes by helping
other users on the mailing lists, reporting bugs or suggesting
features.

**Developer:** a user active in discussions on the developer mailing
list, contributing to the source code, the examples library, or
documentation. Their contributions are added to the source tree by a
committer.

**Committer:** a developer with write access to the source code
repository. Commit access is granted by the Project Management
Committee.

**Project Management Committee (PMC) member:** PMC members control the
direction of the project, based on consensus. The PMC sets the
priorities for the features to be developed, grants and revokes commit
access, and decides what can and cannot be merged into the source tree.

### Project Management Committee

The first PMC was established at the time of the open sourcing of the
initial code and will govern the project. The project has been kick-started 
at BNN, UPC and the PMC memberships reflect this. During the first years, 
the goal of the PMC is to engage with the community, disseminate the model 
and incorporate members from other contributing organizations. After a while,
a PMC based on merit in contributions and demonstration of commitment to the
project will be elected by the committers, from the committers. The
exact procedures will be discussed on the developer mailing list, and
this document will be updated to reflect the outcome of those
discussions when the PMC feels that consensus is reached.

The current members of the PMC are:
-  Miquel Ferriol-Galmés (miquel.ferriol@upc.edu)
-  José Suárez-Varela (jose-rafael.suarez-varela@upc.edu)
-  Albert Cabellos-Aparicio (alberto.cabellos@upc.edu)
-  Pere Barlet-Ros (pbarlet@ac.upc.edu)
