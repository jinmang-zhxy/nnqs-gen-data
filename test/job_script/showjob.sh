nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > /tmp/gpu_usage.csv
nvidia-smi -L > /tmp/gpu_list.csv

printf "%-10s  %-30s  %-10s  %-10s\n" "PID" "FILE" "TIME" "GPUID"
ps -aux | grep -e "[0-9]*:[0-9]* python nnqs.py" -e "nnqs.py" -e "pretrain.py" | awk '{
    file = ""; gpu_id = -1; pid = $2;

    gpu_uuid = "";
    while (("cat /tmp/gpu_usage.csv" | getline line) > 0) {
        split(line, gpu_info, ",");
        if (gpu_info[2] == pid) {
            gpu_uuid = gpu_info[1];
            break;
        }
    }
    close("cat /tmp/gpu_usage.csv");

    if (gpu_uuid != "") {
        while (("cat /tmp/gpu_list.csv" | getline line) > 0) {
            if (index(line, gpu_uuid) > 0) {
                match(line, /GPU ([0-9]+)/, arr);
                gpu_id = arr[1];
                break;
            }
        }
        close("cat /tmp/gpu_list.csv");
    }

    for (i=1; i<=NF; i++) {
        if ($i ~ /--log_file=/) {
            split($i, a, "="); 
            file = a[2]; 
            break;
        }
    }
    printf("%-10d  \033[1m%-30s\033[0m %-10s %-10d\n", $2, file, $10, gpu_id);
}'

rm /tmp/gpu_usage.csv
rm /tmp/gpu_list.csv
