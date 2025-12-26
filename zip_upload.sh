
# zip -r raw_kcd_576p_frustum.zip raw_kcd_576p_frustum/

# dbxcli-linux-amd64 put "raw_kcd_576p_frustum.zip"

# zip -r raw_kcd2_576p_frustum.zip raw_kcd2_576p_frustum/

# dbxcli-linux-amd64 put "raw_kcd2_576p_frustum.zip"

# split -b 20G raw_kcd_576p_frustum.zip part_

# mkdir 2077-12-12
# cd 2077-12-12
# dbxcli-linux-amd64 get "2077/2077-111.mp4" &
# dbxcli-linux-amd64 get "2077/2077-112.mp4" &
# dbxcli-linux-amd64 get "2077/2077-113.mp4" &
# dbxcli-linux-amd64 get "2077/2077-114.mp4" &
# dbxcli-linux-amd64 get "2077/2077-115.mp4" &
# dbxcli-linux-amd64 get "2077/2077-116.mp4" &
# dbxcli-linux-amd64 get "2077/2077-117.mp4" &
# dbxcli-linux-amd64 get "2077/2077-118.mp4" &
# dbxcli-linux-amd64 get "2077/2077-119.mp4" &
# dbxcli-linux-amd64 get "2077/2077-1110.mp4" &
# dbxcli-linux-amd64 get "2077/2077-1111.mp4" &
# dbxcli-linux-amd64 get "2077/2077-1112.mp4" &
# dbxcli-linux-amd64 get "2077/2077-1113.mp4" &
# dbxcli-linux-amd64 get "2077/2077-1114.mp4" &

# wait
# dbxcli-linux-amd64 put raw_osmo_576p_frustum.zip.part_aa &    
# dbxcli-linux-amd64 put raw_osmo_576p_frustum.zip.part_ab &              
# dbxcli-linux-amd64 put raw_osmo_576p_frustum.zip.part_ac &
# dbxcli-linux-amd64 put raw_osmo_576p_frustum.zip.part_ad &

# wait

# zip -r raw_osmo_576p_prompt_1.zip raw_osmo_576p_prompt_1/
# dbxcli-linux-amd64 put "raw_osmo_576p_prompt_1.zip"
# zip -r raw_2077-12-23_576p_prompt_1.zip raw_2077-12-23_576p_prompt_1/
# dbxcli-linux-amd64 put "raw_2077-12-23_576p_prompt_1.zip"
