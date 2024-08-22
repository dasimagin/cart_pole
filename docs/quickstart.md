# Quickstart

## Enviroment
Python is main language of project. So, students may learn control theory and make experiments faster and easier.
Firstly, you need checkout repo and prepare enviroment.

```bash
git clone https://github.com/robotics-laboratory/cart-pole.git
```

We have a built container with all dependencies, use it for development and testing
(you need to have [docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/) installed). Run in root of repo following commands.

Pull latest actual image ```docker compose pull```, that you may start devcontainer in VSCode or run it manually.

```bash
# enter to container
docker exec -it cartpole bash
```

Repo folder is mounted as `/cartpole` dir, so you can edit files in your favorite IDE.
We highly recommend to use `VS Code`, since we provide everything for comfortable development.
After you set up your environment, you can run tests to check that everything is OK.

```bash
pytest tests
```

Also there are some environment variables, which may be useful for you:

- `$CONTAINER_NAME` - name of container (default is `cartpole`)

If you want to use your own python environment, you can install all dependencies manually, using [poetry](https://python-poetry.org/).

```bash
# check poetry config
poetry config --list 

# install all depencies to .venv folder
poetry install

# run tests to check that everithing is OK
poetry run pytest tests
```

## Foxglove
For visualization of real time data we use [foxglove studio](https://foxglove.dev/).
We strongly suggest to use our [instance](http://foxglove.robotics-lab.ru), but you may also setup server with our specific fixes by yourself (more information [here](https://github.com/robotics-laboratory/foxglove)).
In Foxglove Studio select `Open connection` than `Foxglove WebSocket` and enter `ws://localhost:8765` (use your port) in address field.

## Logging
We have convinient logging system, it may show data in real time and replay saved data in [mcap](https://mcap.dev/) format.

```python title="examples/log.py"
--8<-- "examples/log.py"
```

## Simulation
For development and testing of control algorithms, we provide CartPole simulator, which fully implemntet CartPoleBase [interface](/cartpole/common.py). 
The simulation is carried out by numerical integration of parameterized dynamic system (more information [here](/docs/cart_pole.pdf)).
Also simulator may be used to train ML agents.

```python title="examples/simulatio.py"
--8<-- "examples/simulation.py"
```

## Docs
You can build and run docs server locally.

```bash
mkdocs serve -a 0.0.0.0:8000
```
