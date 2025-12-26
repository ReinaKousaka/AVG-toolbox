dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.22 - 12.21.22.54.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.22 - 12.44.15.55.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.22 - 12.54.33.56.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 14.47.53.57.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.00.14.58.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.04.22.59.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.09.23.60.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.23.15.61.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.31.37.62.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 15.56.15.63.mp4" &
dbxcli-linux-amd64 get "2077/1223/Cyberpunk 2077 2025.12.23 - 16.03.27.64.mp4" &
wait

mkdir 2077-12-23

mv "Cyberpunk 2077 2025.12.22 - 12.21.22.54.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.22 - 12.44.15.55.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.22 - 12.54.33.56.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 14.47.53.57.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 15.00.14.58.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 15.04.22.59.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 15.09.23.60.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 15.23.15.61.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 15.31.37.62.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 15.56.15.63.mp4" 2077-12-23/
mv "Cyberpunk 2077 2025.12.23 - 16.03.27.64.mp4" 2077-12-23/

cd 2077-12-23 && rename 's/[^a-zA-Z0-9._-]//g' * && cd ..