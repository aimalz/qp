FROM python:2.7

# Update the OS as needed
RUN apt-get update

# Get TEX fonts for jupyter notebook
RUN apt-get install texlive-latex-recommended -y
RUN apt-get install texlive-latex-extra -y
RUN apt-get install texlive-fonts-recommended -y
RUN apt-get install chktex -y
RUN apt-get install dvipng -y

# Install python requirements using pip:
RUN pip install --upgrade jupyter
RUN pip install nbconvert

# Finally get set up to build the docs
RUN pip install sphinx
RUN pip install sphinx_rtd_theme

# Install the dependencies and the package:
WORKDIR /home
RUN git clone https://github.com/aimalz/qp.git
WORKDIR /home/qp
RUN pip install -r requirements.txt
RUN python setup.py install

# cd to the work directory and launch a jupyter notebook on run
WORkDIR /home/qp/docs/notebooks
CMD jupyter notebook --no-browser --allow-root --port=8888 --ip=0.0.0.0