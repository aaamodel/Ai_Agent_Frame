#!/usr/bin/env bash
# =====================================================================
# 压测编排脚本：一条命令跑完「环境采集 → 真实压测 → mock 压测 → 熔断实测」
#
# 用法：
#   bash benchmark/run_bench.sh                      # 默认 50 并发 / 10 分钟
#   bash benchmark/run_bench.sh 100 10m              # 100 并发 / 10 分钟
#   bash benchmark/run_bench.sh 50 5m --mode mock    # 只跑 mock（零成本）
#   bash benchmark/run_bench.sh --mode both --no-cb  # 不加熔断实测
#
# 环境变量：
#   BENCH_HOST       被测应用地址（默认 http://127.0.0.1:8000）
#   BENCH_MOCK_PORT  mock LLM 端口（默认 8100）
#   BENCH_PY         python 解释器（默认 python）
#   BENCH_LOCUST     locust 可执行（默认 locust）
#   BENCH_CPU/BENCH_MEM  手工填写机器规格（自动探测失败时用）
#
# ⚠️ 诚实红线：本脚本产出的所有数字都是**本机单机环境**数据，
#    不是生产环境数据。报告模板里已内置这句声明，请勿删除。
# =====================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/_results"
mkdir -p "${RESULTS_DIR}"

BENCH_HOST="${BENCH_HOST:-http://127.0.0.1:8000}"
BENCH_MOCK_PORT="${BENCH_MOCK_PORT:-8100}"
BENCH_PY="${BENCH_PY:-python}"
BENCH_LOCUST="${BENCH_LOCUST:-locust}"

# 压测问题集：默认用 benchmark/queries_bench.txt（纯内部依赖）。
# 不要直接复用黄金集 —— tool_cases 里有会触发真实联网搜索 / 写多维表格的问题，
# 它们的延迟会混进 P95，写操作更是会污染真实数据。详见该文件头部说明。
BENCH_QUERIES="${BENCH_QUERIES:-${SCRIPT_DIR}/queries_bench.txt}"
export BENCH_QUERIES

CONCURRENCY="50"
DURATION="10m"
MODE="real"
RUN_CIRCUIT_BREAKER=1

# ---- 参数解析 ----
# 支持两种写法：`--mode mock` 与 `--mode=mock`；剩余位置参数按序视为
# [并发数] [时长]，例如 `bash run_bench.sh 100 10m --mode both`。
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode=*)  MODE="${1#*=}"; shift ;;
    --mode)    MODE="${2:-}"; shift 2 ;;
    --no-cb)   RUN_CIRCUIT_BREAKER=0; shift ;;
    -h|--help)
      sed -n '2,30p' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done
if [ "${#POSITIONAL[@]}" -ge 1 ]; then CONCURRENCY="${POSITIONAL[0]}"; fi
if [ "${#POSITIONAL[@]}" -ge 2 ]; then DURATION="${POSITIONAL[1]}"; fi

case "${MODE}" in
  real|mock|both) : ;;
  *) printf '[run_bench] ❌ --mode 只支持 real / mock / both，收到：%s\n' "${MODE}"; exit 1 ;;
esac

LOG_FILE="${RESULTS_DIR}/run_bench_$(date +%Y%m%d_%H%M%S).log"

hr() { printf '%s\n' "--------------------------------------------------------------------"; }
say() { printf '[run_bench] %s\n' "$*" | tee -a "${LOG_FILE}"; }

say "仓库根目录 : ${REPO_ROOT}"
say "被测地址   : ${BENCH_HOST}"
say "并发/时长  : ${CONCURRENCY} / ${DURATION}"
say "运行模式   : ${MODE}（real=真实模型, mock=mock LLM, both=两轮都跑）"
hr

# =====================================================================
# 0. 工具检查
# =====================================================================
if ! command -v "${BENCH_LOCUST}" >/dev/null 2>&1; then
  say "❌ 找不到 locust（${BENCH_LOCUST}）。安装：pip install locust"
  exit 1
fi
if ! command -v "${BENCH_PY}" >/dev/null 2>&1; then
  say "❌ 找不到 python（${BENCH_PY}）"
  exit 1
fi

# =====================================================================
# 1. 环境采集（报告里必须写清机器规格，否则数字没有意义）
# =====================================================================
ENV_FILE="${RESULTS_DIR}/env.txt"

# 探测 CPU 型号：Linux 读 /proc/cpuinfo，macOS 用 sysctl，都拿不到就用 BENCH_CPU
detect_cpu_model() {
  local model=""
  if [ -r /proc/cpuinfo ]; then
    model="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
  fi
  if [ -z "${model}" ] && command -v sysctl >/dev/null 2>&1; then
    model="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  fi
  if [ -z "${model}" ]; then model="${BENCH_CPU:-unknown}"; fi
  printf '%s' "${model}"
}

detect_cpu_logical() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif [ -n "${NUMBER_OF_PROCESSORS:-}" ]; then
    printf '%s' "${NUMBER_OF_PROCESSORS}"
  else
    printf 'unknown'
  fi
}

detect_mem_mb() {
  if command -v free >/dev/null 2>&1; then
    free -m 2>/dev/null | awk '/^Mem:/{print $2}'
  else
    printf '%s' "${BENCH_MEM:-unknown}"
  fi
}

{
  echo "# 压测环境记录"
  echo "time            : $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host_os         : $(uname -s 2>/dev/null || echo unknown) $(uname -r 2>/dev/null || echo '')"
  echo "python          : $("${BENCH_PY}" --version 2>&1 || echo unknown)"
  echo "locust          : $("${BENCH_LOCUST}" --version 2>&1 | head -n 1 || echo unknown)"
  echo "cpu_logical     : $(detect_cpu_logical)"
  echo "cpu_model       : $(detect_cpu_model)"
  echo "mem_total_mb    : $(detect_mem_mb)"
  echo "git_rev         : $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch      : $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "bench_host      : ${BENCH_HOST}"
  echo "concurrency     : ${CONCURRENCY}"
  echo "duration        : ${DURATION}"
  echo "mode            : ${MODE}"
  echo "note            : 本机单机环境，非生产环境数据"
} > "${ENV_FILE}"
say "环境记录 -> ${ENV_FILE}"
cat "${ENV_FILE}" | tee -a "${LOG_FILE}"
hr

# =====================================================================
# 2. 服务可达性预检
# =====================================================================
say "预检 ${BENCH_HOST}/api/v1/health …"
if command -v curl >/dev/null 2>&1; then
  if curl -sf --max-time 5 "${BENCH_HOST}/api/v1/health" >/dev/null 2>&1; then
    say "✅ 服务可达"
  else
    say "❌ 服务不可达。请先启动应用："
    say "   uvicorn app.main:app --host 127.0.0.1 --port 8000"
    exit 1
  fi
else
  say "⚠️ 没有 curl，跳过预检（压测会直接报错）"
fi

# =====================================================================
# 3. mock 模式：起 mock server
# =====================================================================
MOCK_PID=""
if [ "${MODE}" = "mock" ] || [ "${MODE}" = "both" ]; then
  say "启动 mock LLM 服务（端口 ${BENCH_MOCK_PORT}）…"
  "${BENCH_PY}" "${SCRIPT_DIR}/mock_llm_server.py" \
      --port "${BENCH_MOCK_PORT}" --latency-ms 300 --jitter-ms 100 \
      > "${RESULTS_DIR}/mock_llm.log" 2>&1 &
  MOCK_PID=$!
  sleep 2
  if command -v curl >/dev/null 2>&1 && curl -sf --max-time 5 \
       "http://127.0.0.1:${BENCH_MOCK_PORT}/healthz" >/dev/null 2>&1; then
    say "✅ mock 服务已就绪（pid=${MOCK_PID}）"
  else
    say "❌ mock 服务启动失败，见 ${RESULTS_DIR}/mock_llm.log"
    exit 1
  fi
  say "⚠️ mock 模式要求**应用进程**指向 mock："
  say "   OPENAI_API_BASE=http://127.0.0.1:${BENCH_MOCK_PORT}/v1 uvicorn app.main:app --port 8000"
  say "   若应用已经用真实 key 启动，请重启它，否则这轮会打到真实 API（烧钱且不干净）。"
  hr
fi

cleanup() {
  if [ -n "${MOCK_PID}" ] && kill -0 "${MOCK_PID}" 2>/dev/null; then
    say "停止 mock 服务（pid=${MOCK_PID}）"
    kill "${MOCK_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# =====================================================================
# 4. 跑压测（每个并发档一轮）
# =====================================================================
run_locust() {
  local locust_file="$1"
  local tag="$2"
  local csv="${RESULTS_DIR}/locust_${tag}"
  say "开始压测：${tag}（file=${locust_file} u=${CONCURRENCY} t=${DURATION}）"
  "${BENCH_LOCUST}" -f "${SCRIPT_DIR}/${locust_file}" \
      --headless \
      -u "${CONCURRENCY}" -r "$(( CONCURRENCY / 10 > 0 ? CONCURRENCY / 10 : 1 ))" \
      -t "${DURATION}" \
      --host "${BENCH_HOST}" \
      --csv "${csv}" \
      --only-summary \
      --loglevel WARNING 2>&1 | tee -a "${LOG_FILE}"
  say "CSV 输出：${csv}_stats.csv / ${csv}_failures.csv"
  hr
}

case "${MODE}" in
  real)
    run_locust "locustfile.py" "real_${CONCURRENCY}"
    ;;
  mock)
    run_locust "locustfile_mock.py" "mock_${CONCURRENCY}"
    ;;
  both)
    run_locust "locustfile_mock.py" "mock_${CONCURRENCY}"
    say "⚠️ 接下来跑真实模型压测：请确认应用已切回真实 API（未指向 mock），"
    say "   并确认这轮会消耗真实 token 费用。"
    run_locust "locustfile.py" "real_${CONCURRENCY}"
    ;;
  *)
    say "❌ --mode 只支持 real / mock / both，收到：${MODE}"
    exit 1
    ;;
esac

# =====================================================================
# 5. 熔断降级实测
# =====================================================================
if [ "${RUN_CIRCUIT_BREAKER}" -eq 1 ]; then
  say "开始熔断降级实测（耗时会接近 tier 超时 × 轮次，请耐心等）"
  say "先跑 --dry-run 确认候选数量够降级…"
  "${BENCH_PY}" "${SCRIPT_DIR}/test_circuit_breaker.py" --dry-run 2>&1 | tee -a "${LOG_FILE}"
  hr
  "${BENCH_PY}" "${SCRIPT_DIR}/test_circuit_breaker.py" --repeat 2 --fault hang 2>&1 \
    | tee -a "${LOG_FILE}"
  hr
fi

# =====================================================================
# 6. 汇总
# =====================================================================
say "全部完成。产出物："
say "  环境记录      : ${ENV_FILE}"
say "  压测统计 CSV  : ${RESULTS_DIR}/locust_*_stats.csv"
say "  失败明细 CSV  : ${RESULTS_DIR}/locust_*_failures.csv"
say "  token 统计    : ${RESULTS_DIR}/token_usage.json"
say "  熔断实测报告  : ${SCRIPT_DIR}/熔断切换实测.md"
say "  运行日志      : ${LOG_FILE}"
hr
say "下一步：把上面的数字填进 benchmark/压测报告.md 的表格里。"
say "⚠️ 填表时请保留这句声明：本机单机环境，非生产环境数据。"
