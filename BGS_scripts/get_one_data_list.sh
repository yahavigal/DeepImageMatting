#!/usr/bin/bash


ls -R $1 | awk '
/:$/&&f{s=$0;f=0}
/:$/&&!f{sub(/:$/,"");s=$0;f=1;next}
NF&&f{ print s"/"$0 }' |  grep color | grep -v silhuette | awk 'NR == 1 || NR % 1 == 0' > data_list.txt

