#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
screen -L -S carla_rl .venv/bin/python rl_train.py