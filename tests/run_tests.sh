#!/bin/bash

tests=$(ls | grep '^[A-Z]')
DATE=$(date)
echo $tests
function log_sh() {
    echo "$DATE [TESTS EXECUTION]" $@
}

for i in $tests
do
    python3 $i > /dev/null
    if [ ! $? -eq 0 ]
    then
        log_sh "Test $i failed"
        exit 1
    fi
done
