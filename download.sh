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
# mkdir 2077-12-28
# cd 2077-12-28

# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 17.40.24.07.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 17.46.56.08.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 17.51.56.09.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 17.53.55.10.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 17.58.58.11.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 18.02.55.12.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 20.58.53.13.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 21.04.12.14.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 21.12.48.15.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 22.06.27.26.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 22.12.06.27.mp4" &
# dbxcli-linux-amd64 get "2077/12-28/Cyberpunk 2077 2025.12.28 - 22.17.15.28.mp4" &

# wait
# rename 's/[^a-zA-Z0-9._-]//g' *

dbxcli-linux-amd64 get "sekai-real-walking1_scattered_day.zip.part_aa" &
dbxcli-linux-amd64 get "sekai-real-walking1_scattered_day.zip.part_ab" &
dbxcli-linux-amd64 get "sekai-real-walking1_scattered_day.zip.part_ac" &

wait

cat sekai-real-walking1_scattered_day.zip.part_* > sekai-real-walking1_scattered_day.zip
unzip sekai-real-walking1_scattered_day.zip