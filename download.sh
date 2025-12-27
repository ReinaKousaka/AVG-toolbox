# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.22 - 12.21.22.54.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.22 - 12.44.15.55.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.22 - 12.54.33.56.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 14.47.53.57.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.00.14.58.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.04.22.59.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.09.23.60.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.23.15.61.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.31.37.62.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.56.15.63.mp4" &
# dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 16.03.27.64.mp4" &
# wait

# mkdir 2077-12-23

# mv "Cyberpunk 2077 2025.12.22 - 12.21.22.54.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.22 - 12.44.15.55.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.22 - 12.54.33.56.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 14.47.53.57.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 15.00.14.58.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 15.04.22.59.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 15.09.23.60.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 15.23.15.61.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 15.31.37.62.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 15.56.15.63.mp4" 2077-12-23/
# mv "Cyberpunk 2077 2025.12.23 - 16.03.27.64.mp4" 2077-12-23/

# cd 2077-12-23 && rename 's/[^a-zA-Z0-9._-]//g' * && cd ..
mkdir cityview_2
cd cityview_2


dbxcli-linux-amd64 get "cityview_data_2/1.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/10.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/11.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/12.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/13.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/14.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/15.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/16.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/2.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/3.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/4.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/5.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/6.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/7.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/8.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/9.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_1.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_11.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_12.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_13.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_2.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_3.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_4.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_5.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_7.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_8.MOV" &
dbxcli-linux-amd64 get "cityview_data_2/long_9.MOV" &

wait
