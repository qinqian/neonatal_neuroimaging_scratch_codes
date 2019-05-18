#!/bin/bash -ex

# initial build template ....
#cd training_data
#antsMultivariateTemplateConstruction2.sh -d 3 -i 3 -k 3 -f 4x2x1 -s 2x1x0vox -q 30x20x4 -t SyN -m CC -c 1 -o MY ../template.list

## MUST USE the correct .csv or .txt suffix
#cd test_template/
#antsMultivariateTemplateConstruction2.sh -d 3 -i 3 -k 3 -f 4x2x1 -s 2x1x0vox -q 30x20x4 -t SyN -m CC -c 1 -o MY test.txt

## test again
#cd train2/
#antsMultivariateTemplateConstruction2.sh -d 3 -i 3 -k 3 -f 4x2x1 -s 2x1x0vox -q 30x20x4 -t SyN -m CC -c 0 -o MY ../template2.list

cd training_data
cp ../template.csv .
antsMultivariateTemplateConstruction2.sh -d 3 -i 3 -k 3 -f 4x2x1 -s 2x1x0vox -q 30x20x4 -t SyN -m CC -c 1 -o FUDAN_PVL template.csv

