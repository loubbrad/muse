# Can't get Makefile working, using bash script instead.

# Update and create env
sudo apt-get update && sudo apt-get upgrade
git clone https://github.com/loua19/muse
cd muse
conda create -n muse python=3.10.9
conda activate muse

# Install reqs
pip install -r req.txt

# Download data
mkdir -p data/processed
wget https://www.dropbox.com/s/a82yxmtwypl9aoc/chorale_dataset.json -P data/processed
wget https://www.dropbox.com/s/mmd0xrs4lzywyxg/mutopia.json -P data/processed
