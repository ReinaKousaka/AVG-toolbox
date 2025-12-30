zip -r raw_2077-12-2_camparams.zip raw_2077-12-2_camparams &
zip -r raw_2077-12-12_camparams.zip raw_2077-12-12_camparams &
zip -r raw_2077-12-23_camparams.zip raw_2077-12-23_camparams &
zip -r raw_2077-12-27_camparams.zip raw_2077-12-27_camparams &
wait
dbxcli-linux-amd64 put raw_2077-12-2_camparams.zip &
dbxcli-linux-amd64 put raw_2077-12-12_camparams.zip &
dbxcli-linux-amd64 put raw_2077-12-23_camparams.zip &
dbxcli-linux-amd64 put raw_2077-12-27_camparams.zip &
wait