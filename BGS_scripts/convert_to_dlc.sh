
#1. copy NPE from shared location TBD
#wget or cp or scp to home dir ???????
NPE_ZIP=/home/or/share/snpe-1.8.0.zip

#2. unzip NPE
NPE_NAME=`basename /home/or/share/snpe-1.8.0.zip .zip`
#remove old SNPE
rm -r ~/$NPE_NAME
unzip $NPE_ZIP -d ~/ 

#3. install NPE
pushd ~/$NPE_NAME
sudo apt-get install -y python-dev python-matplotlib python-numpy python-protobuf python-scipy python-skimage python-sphinx wget zip 
source bin/dependencies.sh # verifies that all dependencies are installed
source bin/check_python_depends.sh # verifies that the python dependencies are installed
#find caffe root assuming you have PYTHONPATH env. on your machine
export PYTHONPATH=$OLDPWD/../python
CAFFE_ROOT=`env | grep PYTHONPATH | rev | cut -d: -f1 | rev | cut -d= -f2 | xargs dirname`
sed -i '/CAFFE_ROOT/d' ~/.bashrc 
sed -i '/SNPE_ROOT/d' ~/.bashrc 
echo "export CAFFE_ROOT=$CAFFE_ROOT" >> ~/.bashrc
echo "export SNPE_ROOT=~/$NPE_NAME"  >> ~/.bashrc
bash ~/.bashrc
source ./bin/envsetup.sh -c $CAFFE_ROOT 

#4. convert model 
./bin/x86_64-linux-clang/snpe-caffe-to-dlc -c $1 -b $2 -d $3 --in_layer data --encoding rgba --validation_target cpu snapdragon_820 --strict
popd

#5. upload the converted model to the artifactory
#bash artifactory-deploy.sh -g bgs.intel -a NPE_model  -v 1.0.0 -e dlc  \
#    -d http://artifactoryperc01.iil.intel.com:8081/artifactory/libs-release-local \
#    -u bgs-models -p 1234bgs  -f $3 



