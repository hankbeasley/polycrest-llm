FROM runpod/vscode-server:0.0.0


RUN apt-get update -y && \
    apt-get upgrade -y
    

WORKDIR / 
COPY /src /work/polycrest-llm/src
COPY /.git /work/polycrest-llm/.git

WORKDIR /work/polycrest-llm/src
RUN python -m pip install --upgrade pip
RUN ls
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation
#RUN git pull
RUN python predownload.py

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


#docker build -t hankbeasley/polycess-runpod:0.0.5 -f docker/Dockerfile . 
#docker push hankbeasley/polycress-runpod:0.0.5