import flwr as fl
import argparse
from flwr.server import server_config
import json
import numpy as np
import sys
import traceback
from flwr.common import ndarrays_to_parameters
import torch.optim as optim
from model.ctrgcn import import_class
from model.fedpure import FedPure

reference_frame = np.load('./reference_frame.npy')


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def get_initial_parameters():
    # Instantiate the model
    Model = import_class('model.ctrgcn.Model')
    model = Model(num_class = 60, num_point = 25, num_person = 2,
      graph = 'graph.ntu_rgb_d.Graph', 
      graph_args = dict(labeling_mode='spatial'))
    # Assuming the model has a method `parameters` which returns the model's weights as a list of numpy arrays
    initial_weights = model.parameters()
    initial_parameters = ndarrays_to_parameters(initial_weights)
    return initial_parameters

# Define strategy
# strategy = fl.server.strategy.FedProx(
#     evaluate_metrics_aggregation_fn=weighted_average,
#     min_available_clients=5, min_fit_clients=5, min_evaluate_clients=5, proximal_mu = 0.001
# )

# strategy = fl.server.strategy.FedAvg(
#     evaluate_metrics_aggregation_fn=weighted_average,
#     min_available_clients=2, min_fit_clients=2, min_evaluate_clients=2)

initial_parameters = get_initial_parameters()

strategy = FedPure(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_available_clients=3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    initial_parameters=initial_parameters,
    eta=0.0001,
    eta_l=0.00001,
    tau=1e-9,
    reference_frame = reference_frame
)


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    args = parser.parse_args()

    
    config = server_config.ServerConfig(num_rounds=args.rounds)
    
    # Start Flower server
    hist = fl.server.start_server(
        server_address="localhost:8100",
        config=config,
        strategy=strategy
    )

    np.save('history.npy', hist)
    
if __name__ == "__main__":
    main()
    