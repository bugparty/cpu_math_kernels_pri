#!/bin/bash
pushd sh
pwd
output_log="sieve2_5.o"
rm $output_log
# 提交作业并将输出存储到变量中
#output=$(sbatch sh/sieve2_5.sh)

# 使用 awk 提取任务号
job_id=$(sbatch sieve2_5.sh | awk '{print $4}')

# 打印任务号
echo "Task id is ：$job_id , wait for finish"
# 每隔一秒检查任务状态，直到任务号不再显示
while true; do
    # 检查任务号是否存在于 squeue 输出中
    if ! squeue -u $USER | grep -q "$job_id"; then
        echo "job $job_id finished"
        break
    fi
    # 每隔一秒再次检查
    sleep 1
done
cat ./$output_log
popd

