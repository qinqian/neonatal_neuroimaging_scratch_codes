
# https://stackoverflow.com/questions/15197286/how-can-i-flush-gpu-memory-using-cuda-physical-reset-is-unavailable

sudo fuser -v /dev/nvidia*
sudo kill -9 PID.

