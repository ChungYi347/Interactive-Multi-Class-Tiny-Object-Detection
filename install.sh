sudo bash compile.sh 
pip install torch==1.4.0 torchvision==0.5.0 cycler
pip install -r requirements.txt
sudo python setup.py develop

wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 

sudo apt-get install -y --no-install-recommends dialog swig
cd DOTA_devkit && \
swig -c++ -python polyiou.i && \
python setup.py build_ext --inplac

export PYTHONPATH=`pwd`:$PYTHONPATH