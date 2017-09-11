#!/usr/bin/bash
ls -R $1 | awk '
/:$/&&f{s=$0;f=0}
/:$/&&!f{sub(/:$/,"");s=$0;f=1;next}
NF&&f{ print s"/"$0 }' |  grep silhuette  | awk 'NR == 1 || NR % 5 == 0' > masks.txt 

split -l $[ $(wc -l masks.txt|cut -d" " -f1) * 80 / 100 ] masks.txt
mv xaa masks_train.txt
mv xab masks_test.txt


ls -R $1 | awk '
/:$/&&f{s=$0;f=0}
/:$/&&!f{sub(/:$/,"");s=$0;f=1;next}
NF&&f{ print s"/"$0 }' |  grep color | grep -v silhuette | awk 'NR == 1 || NR % 5 == 0' > data.txt 

split -l $[ $(wc -l masks.txt|cut -d" " -f1) * 80 / 100 ] data.txt
mv xaa data_train.txt
mv xab data_test.txt

cat data_train.txt masks_train.txt > train_list.txt
cat data_test.txt masks_test.txt > test_list.txt 

rm data_train.txt data_test.txt masks_train.txt masks_test.txt data.txt masks.txt
