FROM hankbeasley/vscode-server:NV0.1.3


RUN apt-get update -y && \
    apt-get upgrade -y
    

WORKDIR / 
COPY /src /work/polycrest-llm/src
COPY /.git /work/polycrest-llm/.git

WORKDIR /work/polycrest-llm/src
RUN python -m pip install --upgrade pip
RUN ls
RUN pip install -r requirements.txt
#RUN pip install git+https://github.com/huggingface/transformers
#RUN pip install flash-attn --no-build-isolation
#RUN git pull
#RUN  pip uninstall transformer-engine
#RUN python3.12 -m pip install --upgrade setuptools
#RUN pip install numpy==1.26.4
RUN pip install sagemaker
#RUN pip install --force-reinstall -v "numpy==1.25.2"
#RUN python predownload.py

WORKDIR /
COPY ./docker/post_start.sh .


# Set the working directory inside the container
#WORKDIR /workspace

# # Expose the default port (adjust if necessary)
# EXPOSE 8080

# # Default command (adjust based on your application's requirements)
# CMD ["code-server", "--bind-addr", "0.0.0.0:8080", "."]
EXPOSE 8000
#CMD [ "bin/bash" ]


#docker build -t hankbeasley/polycess-runpod:hb7.0.7 -f docker/Dockerfile . 
#docker push hankbeasley/polycress-runpod:hb7.0.7