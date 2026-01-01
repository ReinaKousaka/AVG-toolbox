zip -r raw_jersey_448p.zip raw_jersey_448p &
zip -r raw_austin_448p.zip raw_austin_448p &
zip -r raw_osmo_448p.zip raw_osmo_448p &
zip -r raw_osmo1_448p.zip raw_osmo1_448p &
zip -r raw_osaka-u_448p.zip raw_osaka-u_448p &
zip -r raw_cityview_2_448p.zip raw_cityview_2_448p &

wait

dbxcli-linux-amd64 put raw_jersey_448p.zip &
dbxcli-linux-amd64 put raw_austin_448p.zip &
dbxcli-linux-amd64 put raw_osmo_448p.zip &
dbxcli-linux-amd64 put raw_osmo1_448p.zip &
dbxcli-linux-amd64 put raw_osaka-u_448p.zip &
dbxcli-linux-amd64 put raw_cityview_2_448p.zip &
wait

