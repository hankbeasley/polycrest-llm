#!/bin/bash

chmod +x /usr/local/bin/start-vscode
chmod +x /usr/local/bin/init
chmod +x /usr/local/bin/serve-remote
chmod +x /usr/local/bin/serve-local

if [ ! -d /workspace/polycrest-llm ]; then
    cp -r /work/polycrest-llm /workspace/polycrest-llm
fi
git config --global user.name "Hank Beasley"
git config --global user.email "hankbeasleymail@yahoo.com"
export HF_TOKEN=$RUNPOD_HF_TOKEN
cd /workspace/polycrest-llm
git pull
start-vscode