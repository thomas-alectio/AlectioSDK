mkdir data
mkdir log
cd data
aws s3 cp s3://alectio-resources/cocosamples . --recursive