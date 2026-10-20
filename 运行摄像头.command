#!/bin/zsh
PPE_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "$PPE_PROJECT_DIR/.venv/bin/python" ]]; then
  PPE_PYTHON="$PPE_PROJECT_DIR/.venv/bin/python"
elif [[ -x "$PPE_PROJECT_DIR/../.venv/bin/python" ]]; then
  PPE_PYTHON="$PPE_PROJECT_DIR/../.venv/bin/python"
else
  print "请先在工程目录建立虚拟环境，并安装 requirements-ppe-lock.txt 中的依赖。"
  read "PPE_REPLY?按回车退出"
  exit 1
fi
cd "$PPE_PROJECT_DIR"
"$PPE_PYTHON" camera_server.py --open-browser
read "PPE_REPLY?服务已结束，按回车退出"
