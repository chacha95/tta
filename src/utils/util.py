# -*- coding: utf-8 -*-
import random


def create_random_port() -> str:
    # open random port for process
    rand = random.randrange(50000, 65353)
    dist_url = f'tcp://127.0.0.1:{rand}'
    return dist_url
