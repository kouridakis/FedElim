# Federated Elimination - An experimental approach to federated learning
The core concept of FedElim is the concept of elimination: whenever the server receives an updated model from a client, its performance is measured and compared to the Current Best Model (CBM). If it is better, the CBM gets eliminated and replaced with the received one. The algorithm works as follows:

0. The server initializes a model and sends it to all clients.

1. Each connected client receives the model, trains it, and sends it back to the server.

2. For each received model:  

    a. The server tests it and compares the results to the CBM. If it is better, the CBM gets eliminated and replaced with the received one.  

    b. The server forwards the received model to all clients, regardless if it was better than the CBM.

3. Repeat from step 1.


The approach of only forwarding models that eliminate the CBM was also considered, however in my testing this resulted in the process quickly hitting a "wall". 


## Benefits
Federated elimination is asynchronous and inherently simple. It does not require selecting a subset of clients for federated training rounds, and it does not require nor await for all clients to answer. These properties make FedElim:

* Resistand to poisoning from malicious clients, as weights which degrade performance cannot replace the CBM.

* Fault tolerant, as clients which disconnect or are slow to respond do not affect the overall training process.

## Installation
The script has been tested on Python 3.10 and the dependencies managed through pipenv. Thus, to set everything up it is recommended to use pipenv and run:
```
pipenv install
``` 
in the root directory of the project.

## Usage
The FashionMNIST dataset is used, which contains images of clothing split into 10 classes. 10 clients are created, each containing the entire set of one class along with additional random elements from other classes, while the server is given the entire testing set provided by the dataset.  

You can adjust the following parameters through command line arguments:
```
-b, --batch_size: The batch size to be used for training (default: 64).
-r, --rounds: The number of training rounds per client (default: 5).
-f, --fail_rate: The probability that a client will fail to return a model (default: 0.5).
-e, --epochs: The number of federated epochs to run (default: 10).
-c, --cutoff: If the received models exceed the cutoff, the server will send only the best model to all clients. Try reducing this number if your device runs out of memory. (default: 1000).
```

```
To execute the script through pipenv, run:
```
pipenv shell
python fed_elim.py
```
