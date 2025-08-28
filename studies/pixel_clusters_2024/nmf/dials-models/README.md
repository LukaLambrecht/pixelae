# Model definition for integration with DIALS

For general information on how to integrate a given model in DIALS, see here:
- [dism-examples](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master?ref_type=heads).
- In particular for this case, the [sklearn NMF example](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master/sklearn_nmf?ref_type=heads) was the main inspiration.
- [dism-cli](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-cli/-/tree/develop?ref_type=heads) (which is the tool used in the examples above).

This folder just contains one particular example, for the pixel NMF models developed in this project.
Integration in DIALS for this projec was achieved in a long back-and-forth interaction with Gabriel (the DIALS developer),
and it is expected that different models, package versions, oprating systems, etc, could give unexpected issues not covered below.
But still, in case they can serve as some useful guidance, I collected some instructions and notes below.

### Define model
This step comes before any DIALS integration code.
It's just a matter of defining a model class, building and training the model (or loading a trained one), and dumping everything into a single neatly packaged `.joblib` file.

See [dism-examples](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master?ref_type=heads) for some minimal examples.
This particular case is mainly based on the [sklearn NMF example](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master/sklearn_nmf?ref_type=heads),
where the model is defined, trained, and dumped to a `.joblib` file in the notebook `main.ipynb`.

For this particular case, the model is defined in a `PixelNMF` class in the file `pixelnmf.py` (in the `mlserver-model` folder).
It is then instantiated and dumped into a `.joblib` file by the `instantiate_pixelnmf.ipynb` notebook.
The loading of the model and evaluation on some example data can be tested with `test_pixelnmf_load.ipynb` and `test_pixelnmf_run.ipynb` respectively.

**Note about dependencies**: the main difficulty with the `.joblib` file produced in this way is that it is not standalone.
It needs the class of which the instance was dumped, but also all its potential dependencies, to be present in the namespace when loading the `.joblib` file.
This is not an issue in most of the `dism-examples` linked above, as they are simple models defined in a single file with no local dependencies.
But on the other hand, the `PixelNMF` class in the file `pixelnmf.py` is not standalone; it relies on many imports of utility classes and functions defined elsewhere in this project.
The problem is that the final packaged model shipped to DIALS only has access to whatever is in the `mlserver-model` folder, nothing outside of it.
There are multiple potential solutions:
- Make all dependencies into packages that can be imported without needing to specify the path. Recommended by Gabriel but not yet tried, likely too much overhead and overcomplication.
- Copy all dependencies to the `mlserver-model` folder (and change all import paths to flat local imports). Obviously not ideal in case of future modifications, but chosen for now. 

### Prepare required handling and configuration files
A couple of files need to be prepared to parse the model into a format suitable for DIALS.
In particular:
- `template.yaml`, specifying some configuration settings such as input monitoring element names, shapes, and data types.
- `app.py` (inside the `mlserver-model` folder), specifying the interface between DIALS syntax and custom model syntax.

The versions in this particular case were copied from the [sklearn NMF example](https://gitlab.cern.ch/cms-ppd/technical-support/tools/dism-examples/-/tree/master/sklearn_nmf?ref_type=heads),
and modified as needed (e.g. specify multiple monitoring elements, etc).

### Get the dismcli tool
The tool that parses the model into DIALS (using the config files defined in the previous step), is called DIALS Inference Service Manager (DISM).

It is installable via pip in a python virtual environment. Follow the steps below:
```
python3.11 -m venv venv
source venv/bin/activate
pip install "cmsdism[docker,podman]"
```

**Note about python version**: Only tested with python 3.11 (following Gabriel's instructions), not clear how sensitive everything is to the precise python version.
If you don't have python 3.11, you can install it (on your local pc) with `sudo apt intall python3.11`.
You might also need to do `sudo apt install python3.11-venv` in order to be able to create virtual environments as shown above.

**Note about where to run this**: In principle, dismcli could be installed and run on lxplus.
But in that case, you need to modifiy `pip install "cmsdism[docker,podman]"` to `pip install "cmsdism[podman]"`, the inclusion of docker seems to mess up things.
Also, running it on `/eos` seems not to work, somehow the correct paths cannot be mounted inside the container (see next steps); you need to run from `/afs`.
However, for the particular case of the mlserver (e.g. for sklearn models), there are other issues on lxplus for which no solution currently is known.
So you probably need to run on your local pc.
(This issue is not there for the triton server, e.g. for tensorflow/onnx models).

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
