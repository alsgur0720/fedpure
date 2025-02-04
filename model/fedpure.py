from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.fedopt import FedOpt
from detector import Data_poisoning_attack_detector

# pylint: disable=line-too-long
class FedPure(FedOpt):
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
        reference_frame: Tuple,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=0.0,
            beta_2=0.0,
            tau=tau,
            lambda_mal = 0,
        )

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdagrad(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        transform_prototype: List,
        velocity: List,
        reference_frame: Tuple,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        if server_round == 1:
            self.current_weights = fedavg_weights_aggregate
            self.lambda_mal = 0
        
        
        else :
            
            self.lambda_mal = Data_poisoning_attack_detector(transform_prototype, velocity, reference_frame)
            
            if self.lambda_mal:
            
                delta_t: NDArrays = [
                    x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
                ]

                # m_t
                if not self.m_t:
                    self.m_t = [np.zeros_like(x) for x in delta_t]
                self.m_t = [
                    np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
                    for x, y in zip(self.m_t, delta_t)
                ]

                # v_t
                if not self.v_t:
                    self.v_t = [np.zeros_like(x) for x in delta_t]
                self.v_t = [x + np.multiply(y, y) for x, y in zip(self.v_t, delta_t)]

                new_weights = [
                    x + self.eta * y / (np.sqrt(z) + self.tau)
                    for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
                ]

                self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated
