#!/usr/bin/bash


ls -R $1 | awk '
/:$/&&f{s=$0;f=0}
/:$/&&!f{sub(/:$/,"");s=$0;f=1;next}
NF&&f{ print s"/"$0 }' |  grep color | grep -v silhuette | awk 'NR == 1 || NR % 2 == 0' > data.txt 

if  [ "$2" == "--real" ]
then
    mv data.txt data_real.txt
    return 0
fi    
split -l $[ $(wc -l data.txt|cut -d" " -f1) * 80 / 100 ] data.txt
mv xaa data_train.txt
mv xab data_test.txt

cat data_train.txt > train_list.txt
cat data_test.txt  > test_list.txt 

rm data_train.txt data_test.txt data.txt 
