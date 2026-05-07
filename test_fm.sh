#!/bin/bash

# ======================== 批量配置（仅需修改这里） ========================
# 外层循环：需要执行的模型名称列表（根据你的实际模型名修改）
MODEL_NAMES=(
    "sft_random"
)


# ======================== 两层循环执行原有逻辑 ========================
# 外层：遍历所有模型名称
for MODELNAME in "${MODEL_NAMES[@]}"; do
    
    echo -e "\n=================================================="
    echo "开始执行：模型名=${MODELNAME}"
    echo "=================================================="

  
    SERVER_CMD="python -m sglang.launch_server \
    --model-path /data/sty/sty-lmf/saves/qwen3-0.6b/${MODELNAME} \
    --tokenizer-path /data/sty/sty-lmf/saves/qwen3-0.6b/${MODELNAME} \
    --dtype bfloat16 \
    --port 30000 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.7 \
    --trust-remote-code \
    --context-length 21000 \
    --schedule-conservativeness 0.3 \
    --cuda-graph-max-bs 768 \
    --chunked-prefill-size 4096 \
    --dp 4"

    # 定义你要运行的 Python 文件路径（请替换成实际路径）
    PYTHON_SCRIPT="/data/sty/sty-lmf/sglang_inference_fm.py"

    # 1. 启动 SGLang 服务（后台运行，并记录进程ID）
    echo "=== 启动 SGLang 服务 ==="
    mkdir -p testlog
    $SERVER_CMD > testlog/sglang_server_${MODELNAME}.log 2>&1 &
    SERVER_PID=$!
    echo "SGLang 服务已启动，进程ID: $SERVER_PID，日志文件: sglang_server_${MODELNAME}.log"

    # 2. 等待服务启动完成（可选：根据实际情况调整等待时间）
    echo "=== 等待服务加载完成 ==="
    sleep 180  # 模型加载需要时间，根据你的模型大小调整（比如4B模型建议至少等30秒）

    # 3. 检查服务是否正常启动
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "错误：SGLang 服务启动失败，请查看 sglang_server_${MODELNAME}.log 日志"
        exit 1
    fi

    # 4. 运行目标 Python 文件
    echo "=== 开始运行 Python 文件: $PYTHON_SCRIPT ==="
    python $PYTHON_SCRIPT --modelname ${MODELNAME}
    PYTHON_EXIT_CODE=$?

    # 5. 关闭 SGLang 服务
    echo "=== 关闭 SGLang 服务 ==="
    if ps -p $SERVER_PID > /dev/null; then
        kill $SERVER_PID
        # 等待进程退出（可选）
        sleep 5
        # 如果进程没被杀掉，强制终止
        if ps -p $SERVER_PID > /dev/null; then
            kill -9 $SERVER_PID
            echo "已强制终止 SGLang 服务"
        else
            echo "SGLang 服务已正常关闭"
        fi
    else
        echo "SGLang 服务已提前退出"
    fi

    # 6. 退出脚本（继承 Python 文件的退出码）
    # 注意：这里将原有的 exit 改为仅记录退出码，避免循环提前终止
    CURRENT_EXIT_CODE=$PYTHON_EXIT_CODE
    # ======================== 你原有脚本内容结束 ========================

    # 容错处理：当前任务失败时提示，可选择是否继续
    if [ ${CURRENT_EXIT_CODE} -ne 0 ]; then
        echo -e "\n⚠️  模型${MODELNAME}执行失败，退出码: ${CURRENT_EXIT_CODE}"
        echo "是否继续执行下一个任务？(y/n，默认n)"
        read -r ANSWER
        if [ "${ANSWER}" != "y" ]; then
            echo "❌ 手动终止批量执行"
            exit ${CURRENT_EXIT_CODE}
        fi
    fi

    # 组间间隔：清理显存，避免连续加载模型导致OOM
    echo -e "\n等待10秒后执行下一个任务..."
    sleep 10
done

echo -e "\n🎉 所有模型的任务都已执行完成！"
exit 0