# USU LLM Class

Repository for USU LLM Class DSAI-5810/6810 - hw11 frontend client

## Contributing

### Set up your environment

Install `uv` using the instructions found here: <https://docs.astral.sh/uv/getting-started/installation/>

### Add dependencies

If you need to add dependencies to run your code, for example cowsay, you can do so with:

```bash
uv add cowsay
```

### Change the code

Update the code to complete your homework. You can run your experiment like so:

```bash
uv run main.py
```

If you prefer working in a jupyter notebook setting you can run:

```bash
uv run --with jupyter jupyter lab
```

### Run formatting/linting/tests

```bash
uv run ruff format
uv run ruff check
uv run pytest
```

### ZenML

Create your server with:

```bash
docker run -it -d -p 8080:8080 --name zenml zenmldocker/zenml-server
```

You will need to visit the ZenML dashboard at http://localhost:8080 and activate the server by creating an initial admin user account. You can then connect your client to the server with the web login flow:

```bash
zenml login http://localhost:8080
```

You can stop it and start it back up again with:

```bash
docker stop zenml
docker start zenml
```

To delete the server and start over you can run:

```bash
docker rm zenml
```
