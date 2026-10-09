# 用 Anaconda 独立环境运行

本项目使用名为 `rl_carracing` 的 Conda 环境，Python 3.11。无需再激活项目原有的 `.venv`。

## 从零创建与安装

打开 **Anaconda Prompt**，逐行执行；如果环境已创建并安装成功，跳到下一节。

```bat
conda create -n rl_carracing python=3.11 pip -y
conda activate rl_carracing
cd /d D:\Desktop\RL_project
python -m pip install pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 swig==4.3.0
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python -m pytest -q
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print('CUDA:',torch.cuda.is_available())"
```

Box2D 编译需要 Visual Studio C++ Build Tools；本机之前已经成功编译。SWIG 安装在同一个 Conda 环境内；激活环境后再安装依赖，并保留 `--no-build-isolation`。

## 每次开始实验

在 **Anaconda Prompt** 中：

```bat
conda activate rl_carracing
cd /d D:\Desktop\RL_project
python -m racing.suite --preset smoke --device cuda --out runs_conda
```

确认短流程通过后运行试验或正式实验：

```bat
python -m racing.suite --preset pilot --device cuda --out runs_conda
python -m racing.suite --preset main --device cuda --out runs_conda
```

三个命令分别对应自检、10 万决策试验、正式多种子实验，不必每次都重复自检。`--out runs_conda` 避免与之前 `.venv` 的验证结果目录冲突；同名已完成实验不允许覆盖，可换输出目录，或按主 README 使用 `--skip-completed`。

单独运行某个算法：

```bat
python -m racing.train --algo ppo --steps 1000000 --device cuda --out runs_conda_single
```

把 `ppo` 换成 `a2c` 或 `vpg50` 即可。其余命令见 README：把 `.\.venv\Scripts\python.exe` **统一换成 `python`**，并使用实际输出目录。

完成后退出环境：

```bat
conda deactivate
```

## 在普通 PowerShell 中运行

不修改全局 shell 配置也能执行：

```powershell
cd D:\Desktop\RL_project
& D:\anaconda\Scripts\conda.exe run --no-capture-output -n rl_carracing python -m racing.suite --preset pilot --device cuda --out runs_conda
```

如果 PowerShell 已配置 Conda，则直接 `conda activate rl_carracing`，再用 `python` 即可。`cd /d` 是 Anaconda Prompt / cmd 的写法，PowerShell 只用 `cd`。

## 保存环境版本

在 Anaconda Prompt 中：

```bat
conda activate rl_carracing
cd /d D:\Desktop\RL_project
conda env export --no-builds > environment-conda-local.yml
python -m pip freeze > requirements-resolved-conda.txt
```

跨电脑重建优先采用上面的显式创建和安装步骤，CUDA PyTorch 安装源也一并保留。导出的本机 YAML 可能包含本机 prefix 路径，不能不加检查地用于其他电脑。
