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

Build: `dismcli build -f template.yaml -u https://dev-cmsdials-api.web.cern.ch`.

Test: `dismcli start-api -r PixelNMF` (change the name to the name of your model).
If this runs correctly, it will print out a whole bunch of stuff, and at some point just seem to hang.
This is the expected behaviour, it means the server is running (locally).
Open another terminal (and go to this directory and start the virtual environment), and run `python test_predictions.py -p 8080 -n pixel-barrel-nmf`.
You will need to update this script (and the command line args) for your particular use case though.
In case it runs correctly, you should get the expected output; else you can check the terminal window where the server is running for the particular error message.
You can stop the server with ctrl+Z, but you also need to do `docker container ls` and `docker rm -f <id>` to free up the port.

Finally, do `dismcli package -u https://dev-cmsdials-api.web.cern.ch`.

### Various notes
- You can remove the hidden directory `.dism-artifacts` to start from scratch. This is similar in concept to deleting cashes or cookies if they give issues.
- For some reason, the docker image specified in the example `template.yaml` (namely `registry.cern.ch/docker.io/seldonio/mlserver:1.5.0`) did not work for me because of something that seemed like a denied permission issue. Instead, you can use the following one: `seldonio/mlserver:1.5.0`. Upon further discussion with Gabriel, it is really better to use the `registry.cern.ch` version of the image in DIALS. To make this work in local testing, you can first pull the `seldonio/mlserver:1.5.0` image, and then run `docker tag seldonio/mlserver:1.5.0 registry.cern.ch/docker.io/seldonio/mlserver:1.5.0`. Then you can use the `registry.cern.ch/docker.io/seldonio/mlserver:1.5.0` image in the `template.yaml` file, and it will locally redirect to the `seldonio/mlserver:1.5.0` image. When actually uploaded to DIALS, there should be no problem with accessing the `registry.cern.ch` version of the image.
