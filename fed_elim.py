import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Any
import random
import copy

random.seed(42)

# Prepare device
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device}")


# Model taken from the pytorch quickstart tutorial
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Global functions
def get_loss_fn():
    return nn.CrossEntropyLoss()

def get_optimizer(model: nn.Module):
    return torch.optim.SGD(model.parameters(), lr=1e-3)

# Global variable that holds whichever model is currently being used.
current_model = NeuralNetwork().to(device)


def red(s: str):
    return '\033[91m' + s + '\033[0m'

def green(s: str):
    return '\033[92m' + s + '\033[0m'


class Client:
    def __init__(self, data: DataLoader):
        self.data = data
    
    def process(self, model_state: dict[str, Any], rounds: int = 1, fail_rate: float = 0) -> dict[str, Any] | None:
        """
        Trains the model on the client data and returns the updated model.

        Args:
            model_state: The state_dict of the model to be trained.
            rounds: The number of training rounds to perform.
            fail_rate: The probability that the client will "disconnect" and not return a model.
        
        Returns:
            A deep copy of the trained model's state_dict if the client does not "disconnect",
            or None if the client "disconnects".
        """
        if random.random() >= fail_rate:
            print(f"{red('d/c')}", end=" ")
            return None

        current_model.load_state_dict(model_state)
        loss_fn = get_loss_fn()
        optimizer = get_optimizer(current_model)

        for _ in range(rounds):
            current_model.train()
            loss: Any
            
            for batch, (X, y) in enumerate(self.data):
                X, y = X.to(device), y.to(device)

                pred = current_model(X)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        loss = loss.item() # type: ignore
        print(f" {loss:>7.4f}", end=" ")
        return copy.deepcopy(current_model.state_dict())

class Server:
    def __init__(self, test_data: DataLoader):
        self.test_data = test_data
        self.best_model = copy.deepcopy(NeuralNetwork().to(device).state_dict())
        self.best_accuracy = -1
    
    def process(self, client_state) -> bool:
        """
        Tests the received model from a client and compares it to the current best model.
        If the received model is better, it replaces the current best model.

        Args:
            client_state: The state_dict of the model received from a client.

        Returns:
            True if the received model is better than the current best model.
        """
        current_model.load_state_dict(client_state)
        loss_fn = get_loss_fn()
        current_model.eval()

        size = len(test_dataloader.dataset) # type: ignore
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = current_model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        accuracy = 100*correct
        print(f"Avg loss:{test_loss:>7.4f} | Accuracy:{accuracy:>5.1f}% |", end=" ")
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = copy.deepcopy(current_model.state_dict())
            return True
        else:
            return False
        

# Download the FashinMNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]


# Hyperparameters
batch_size = 64
client_rounds = 5
fail_rate = 0.5
epochs = 10
model_cutoff = 1000 # this is for time and memory constraint reasons


# The FashionMNIST dataset contains 10 classes, each corresponding to a type of clothing.
# We will create 10 clients, each of which will contain all elements of one class and a random selection of other elements.
clients: list[Client] = []
for i in range(10):
    print(f"Preparing data for Client #{i+1:<2}: {f'({classes[i]})':<13}...", end=" ")
    client_data = [x for x in training_data if x[1] == i]
    for _ in range(random.randint(1000, 2000)):
        x = random.randint(0, len(training_data)-1)
        client_data.append(training_data[x])
            
    clients.append(Client(DataLoader(client_data, batch_size=batch_size))) # type: ignore
    print(f"ready with {len(client_data)} samples.")


test_dataloader = DataLoader(test_data, batch_size=batch_size)
server = Server(test_dataloader)


server_queue: list[dict[str, Any]] = []
client_queues: list[list[dict[str, Any]]] = [
    [copy.deepcopy(current_model.state_dict())] for _ in range(10)
]


# Federated learning
processed_models = 0
for epoch in range(epochs):
    temp_models = processed_models
    temp_accuracy = server.best_accuracy
    cutoff = False
    print(f"\nEpoch {epoch+1:>3}/{epochs}")
    print("-" * 90)

    print(f"Running clients...")    
    for i, client in enumerate(clients):
        print(f"\nClient #{i+1:<2}: {len(client_queues[i])} models in queue.")
        for j, model_state in enumerate(client_queues[i]):
            response = client.process(model_state, client_rounds, fail_rate)
            if response is not None:
                server_queue.append(response)
            if (j+1) % 10 == 0:
                print("")
        # Clear the queue
        client_queues[i] = []
    
    processed_models += len(server_queue)
    cutoff = len(server_queue) >= model_cutoff

    print("\nRunning server...")
    if cutoff:
        print(f"Total queue exceeded cutoff ({model_cutoff}), will send best model to clients.")

    for i, received_model in enumerate(server_queue):
        print(f"Model #{i+1:<5} |", end=" ")
        if server.process(received_model):
            print(green("ACCEPTED"))
        else:
            print("")
        
        if not cutoff:
            for queue in client_queues:
                queue.append(copy.deepcopy(current_model.state_dict()))
    
    if cutoff:
        for queue in client_queues:
            queue.append(copy.deepcopy(server.best_model))
    # Clear the queue
    server_queue = []

    print("-" * 90)
    print(f"Total models processed: {processed_models},", end=" ")
    print(green(f"+{processed_models-temp_models}") + " this epoch.")
    print(f"Best accuracy: {server.best_accuracy:.2f}%,", end=" ")
    print(green(f"+{server.best_accuracy-temp_accuracy:.2f}%") + " this epoch.")
