# RoboOrchard Core

**robo_orchard_core** is the core package of the project **RoboOrchard**, which provides the basic infrastructure for the framework, such as configuration management, data structure, environment abstraction, etc.

## Features

TBC

## Getting started

### Requirements

* Python 3.10

### Installation

#### From pip

```bash
pip install robo_orchard_core
```

#### From source

```bash
cd /path/to/robo_orchard_core
make install
```


## Contribution Guide

### Install by editable mode

```bash
make install-editable
```

### Install development requirements

```bash
make dev-env
```

### Lint

```bash
make check-lint
```

### Auto format

```bash
make auto-format
```

### Type checking

```bash
pyright
```

### Build docs

```bash
make docs
```

### Run test

```bash
make test
```

## Acknowledgement

Our work is inspired by many existing robotics frameworks, such as [OpenAI Gym](https://gym.openai.com/), [Robosuite](https://robosuite.ai/docs/index.html), [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html), [PyTorch3D](https://github.com/facebookresearch/pytorch3d) etc. We would like to thank all the contributors of all the open-source projects that we use in RoboOrchard.
