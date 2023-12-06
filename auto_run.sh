#!/bin/bash

data_list=(0 1 2 3)
#data_list=(1)
#delay_list=(1 2 3 4)
delay_list=(1)
#method_list=(ours base)
method_list=(ours)



for ((i=0; i<${#data_list[@]}; i++))
do
    for ((j=0; j<${#delay_list[@]}; j++))
    do
	for ((k=0; k<${#method_list[@]}; k++))
        do
	    data=${data_list[i]}
            delay=${delay_list[j]}
	    method=${method_list[k]}
            echo data: ${data}, delay: ${delay}, method: ${method}
	    #echo -e "\n"
            LOGFILE=./logs/${data}-${delay}-${method}.log
            python3 test.py --data ${data} --delay ${delay} --method ${method} > ${LOGFILE} 2>&1
	done
    done
done


echo "all done"


