#!/bin/bash
# Dependencies
apt-get -y update
apt-get -y install git
apt-get -y install sudo
apt upgrade
apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt install python3.8 -y
apt install -y python3.8-distutilsapt insta
apt install -y wget
apt-get install -y libopenblas-dev
apt-get install psmisc

# Install Debona
cd ~
git clone https://github.com/ChristopherBrix/Debona.git
cd Debona/
./install_tool.sh v1