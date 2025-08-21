# Model definition for integration with DIALS

For general information on how to integrate a given model in DIALS, see here:
- [dism-examples](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master?ref_type=heads).
- In particular for this case, the [sklearn NMF example](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master/sklearn_nmf?ref_type=heads) was the main inspiration.
- [dism-cli](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-cli/-/tree/develop?ref_type=heads) (which is the tool used in the examples above).

This folder just contains one particular example, for the pixel NMF models developed in this project.
But in case they can serve as some useful guidance, follow the steps below
(preliminary, just some notes as I struggle along, to be updated with more final info!).

### Define model
This step comes before any DIALS integration code;
it's just a matter of defining a model class,
building and training the model (or loading a trained one),
and dumping everything into a single neatly packaged `.joblib` file.

See [dism-examples](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master?ref_type=heads) for some minimal examples.
This particular case is mainly based on the [sklearn NMF example](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master/sklearn_nmf?ref_type=heads),
where the model is defined, trained, and dumped to a `.joblib` file in the notebook `main.ipynb`.

For this particular case, the model is defined in a `PixelNMF` class in the file `pixelnmf.py`.
It is then instantiated and dumped into a `.joblib` file by the `instantiate_pixelnmf.ipynb` notebook.

Note: the `PixelNMF` class in the file `pixelnmf.py` is not standalone;
it contains many imports of utility classes and functions defined elsewhere in this project.
Consequently, the `.joblib` file by itself is pretty useless;
to be able to load and actually use the model somewhere in another script,
the `PixelNMF` class and all its dependencies need to be importable.
Not sure if this will be a problem for packaging the model and shipping it to DIALS,
this remains to be seen.
It is more complicated than the example linked above, where the entire model is defined in a single class/file,
with no other local dependencies (only external packages such as `sklearn`).

### Prepare required handling and configuration files
A couple of files need to be prepared to parse the model into a format suitable for DIALS.
In particular:
- `template.yaml`.
- `mlserver-model` and all files in it.

The versions in this particular case were copied from the [sklearn NMF example](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master/sklearn_nmf?ref_type=heads),
and slightly modified as needed (e.g. specify multiple monitoring elements, etc).

More detailed information to write when it is actually validated and working.

Note: in this particular case, the `CodeUri` and `ModelUri` start with `/data`.
This is needed for making it work with a Docker container, see below for more info. 

### Get the dismcli tool
The tool that parses the model into DIALS (using the config files defined in the previous step),
is called DIALS Inference Service Manager (DISM).

An installation script is provided [here](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/blob/master/scripts/install_dismcli.sh?ref_type=heads),
but the only strictly required steps are the `wget` and `tar` commands.
All other steps are needed just to be able to call `dismcli` from anywhere in the terminal,
but optionally you can skip these and just always call the executable using the correct path
(between where it was installed and where you currently are in the terminal,
e.g. `../dismcli/dismcli` instead of `dismcli`).

### Run the dismcli tool
Follow the steps in the [sklearn NMF example](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master/sklearn_nmf?ref_type=heads).

Note: the `dismcli` tool seems to be incompatible with `lpxlus` or the SWAN terminal,
so it needs to be installed and run locally on your machine.
This also implies your model needs to be able to run locally,
which is a little annoying if the whole project was developed on SWAN until now,
it could potentially be a big hassle with getting the environment right,
not sure if there is a better approach.

Note: if your local machine is running a relatively old OS (such as mine),
`dismcli` might not run on your machine; it will crash with an error of which the essential part is
`/lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.38' not found`.
In that case, a workaround exists using a docker container, see below.

### Using a Docker container to run the dismcli tool
This is optional and only needed if the `dimscli` tool crashes without a container
(e.g. because of a too old operating system on your computer).

Make a Dockerfile with the following contents:

```
# build with (from this directory): docker build -t dismcli-env .
# run with: docker run --rm --it dismcli-env


FROM ubuntu:24.04

# Make sure things can build/run
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libpython3.12 \
    && rm -rf /var/lib/apt/lists/*

# Copy your program into the container
WORKDIR /app
COPY ./dismcli/ /app/dismcli/

# Set environment so it uses the right Python lib
ENV LD_LIBRARY_PATH=/app/dismcli/_internal:$LD_LIBRARY_PATH

# Default to an interactive shell
CMD ["/bin/bash"]
```

The folder structure seems to matter;
put this Docker file at the same level as the `dismcli` folder with the `dismcli` installation.

You can build the container with `docker build -t dismcli-env .`
and then run it with `docker run --rm -it -v /home/luklambr/Programs/pixelae:/data/pixelae dismcli-env`.
The `-v` argument is to mount a local directory to the `/data` directory in the container
(required to have access to the model definition inside the container).

Inside the container, run
```
./dismcli/dismcli build -f /data/pixelae/studies/pixel_clusters_2024/nmf/dials-models/template.yaml -u https://dev-cmsdials-api.web.cern.ch
```

and then
```
./dismcli/dismcli package -u https://dev-cmsdials-api.web.cern.ch
```

and finally copy the output the mounted directory (for accessing it outside the container)
with `cp -r .dism-artifacts/ /data/pixelae/studies/pixel_clusters_2024/nmf/dials-models/`

Note: the strongly recommended step `./dismcli/dismcli start-api` cannot yet be run in this way,
as it requires a docker container within a docker container (which might or might not be possible but doesn't seem like a good idea).
A workaround remains to be found.
