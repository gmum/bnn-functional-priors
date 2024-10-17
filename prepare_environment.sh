#!/bin/bash

echo "Preparing python environment..."
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
git clone https://github.com/tkusmierczyk/reparameterized_pytorch.git
mv reparameterized_pytorch/reparameterized bnngp/
rm -r reparameterized_pytorch
echo "Done. Turn it on with: source .env/bin/activate"
