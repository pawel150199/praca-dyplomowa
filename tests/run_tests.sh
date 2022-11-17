#!/bin/bash

tests=$(ls | grep '^[A-Z]')
DATE=$(date)

function log_sh() {
    echo "[$DATE]  [TESTS EXECUTION]" $@
}

for i in $tests
do
    python3 $i 1>&2 > /dev/null
    if [ ! $? -eq 0 ]
    then
        log_sh "Test $i failed!"
        exit 1
    else
        log_sh "Test $i passed!"
    fi
done
