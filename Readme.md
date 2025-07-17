# Prototype Correlation Fusion-based Attack Detection and Purification for Fedrated Skeleton-based Action Recognition


## Environment

- The code is developed with CUDA 12.2, ***Python >= 3.10.0***, ***PyTorch >= 2.0.0***

    0. [Optional but recommended] create a new conda environment.
        ```
        conda create -n fedpure python=3.10.0
        ```
        And activate the environment.
        ```
        conda activate fedpure
        ```

    1. Install the requirements
        ```
        pip install -r requirements.txt
        ```



The commands are as follows.

```
# Prepare for Federated Learning

python prepare_fed.py -- round 300

# Train federated skeleton-based action recognition

CTR-GCN + FedPure

python main_federated.py --config ./config/default.yaml --work-dir "your_work_directory" --phase train --save-score True --device "your_device_number" --num_clients 10 --client_id 0

 Modify the dataset path in your configuration file.

Setting the num-client argument to 10 and running client_id from 0 to 9 will train a total of 10 clients.


STGCN + FedPure

python main_federated.py --config ./config/default_stgcn.yaml --work-dir "your_work_directory" --phase train --save-score True --device "your_device_number" --num_clients 10 --client_id 0

 Modify the dataset path in your configuration file.

Setting the num-client argument to 10 and running client_id from 0 to 9 will train a total of 10 clients.


HD-GCN + FedPure

python main_federated.py --config ./config/default_hdgcn.yaml --work-dir "your_work_directory" --phase train --save-score True --device "your_device_number" --num_clients 10 --client_id 0

 Modify the dataset path in your configuration file.

Setting the num-client argument to 10 and running client_id from 0 to 9 will train a total of 10 clients.

```



