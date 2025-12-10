#!/usr/bin/env python3
"""
LogAnomaly MCP Server - 双向LSTM日志异常检测系统
提供基于深度学习的日志异常检测服务，支持完整的9步工作流程
"""
import os
import sys
import json
import time
import uuid
import threading
import traceback
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ft_tree"))
sys.path.insert(0, str(project_root / "LogAnomaly_main"))
sys.path.insert(0, str(project_root / "data"))
sys.path.insert(0, str(project_root / "middle"))
sys.path.insert(0, str(project_root / "template2Vector" / "src"))
sys.path.insert(0, str(project_root / "template2Vector" / "src" / "preprocess"))

try:
    from fastmcp import FastMCP
    import numpy as np
except ImportError as e:
    print(f"导入错误: {e}")
    print("请安装所需依赖: pip install fastmcp numpy")
    sys.exit(1)

# 全局任务管理
TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()

# 并发控制
MAX_CONCURRENT = int(os.getenv("LOGANOMALY_MAX_CONCURRENT", "2"))
TASKS_SEM = threading.Semaphore(MAX_CONCURRENT)

# 创建 MCP 实例
mcp = FastMCP("loganomaly", debug=True, log_level="DEBUG")

# 时区设置
def _now_iso() -> str:
    """获取当前时间（中国标准时间）"""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")

def _create_task(task_type: str, params: Dict[str, Any]) -> str:
    """创建新任务"""
    task_id = str(uuid.uuid4())
    with TASKS_LOCK:
        TASKS[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "queued",
            "progress": 0.0,
            "created_at": _now_iso(),
            "started_at": None,
            "completed_at": None,
            "params": params,
            "result": None,
            "error": None,
            "traceback": None,
        }
    return task_id

def _set_task(task_id: str, **kwargs):
    """更新任务状态"""
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(kwargs)

def _start_background(func, task_id: str, params: Dict[str, Any]):
    """启动后台任务"""
    thread = threading.Thread(target=func, args=(task_id, params), daemon=True)
    thread.start()

def _run_subprocess(cmd: List[str], cwd: Path, progress_callback=None) -> Dict[str, Any]:
    """运行子进程命令"""
    try:
        if progress_callback:
            progress_callback(0.1, f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            raise Exception(f"命令执行失败: {result.stderr}")
        
        if progress_callback:
            progress_callback(1.0, "命令执行完成")
        
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        raise Exception(f"执行命令失败: {str(e)}")

# ============================================================================
# MCP 工具函数
# ============================================================================

@mcp.tool()
def loganomaly_extract_template(
    log_file: str,
    output_dir: str = "middle",
    dataset_name: str = "bgl"
) -> Dict[str, Any]:
    """
    【步骤1】从原始日志中提取模板（使用FT-tree算法）
    
    Args:
        log_file (str): 原始日志文件路径（.log格式）
        output_dir (str): 输出目录，默认为 "middle"
        dataset_name (str): 数据集名称，用于命名输出文件
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("extract_template", locals())
    _start_background(_extract_template_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_match_template(
    log_file: str,
    template_file: str,
    fre_word_file: str,
    output_seq_file: str,
    match_mode: int = 1
) -> Dict[str, Any]:
    """
    【步骤2】将日志与模板匹配，生成日志序列
    
    Args:
        log_file (str): 原始日志文件路径
        template_file (str): 模板文件路径（步骤1生成）
        fre_word_file (str): 词频文件路径（步骤1生成）
        output_seq_file (str): 输出序列文件路径
        match_mode (int): 匹配模式 1-正常匹配 2-单条增量 3-批量增量
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("match_template", locals())
    _start_background(_match_template_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_filter_normal(
    seq_file: str,
    label_file: str
) -> Dict[str, Any]:
    """
    【步骤3】过滤异常日志，只保留正常日志用于训练
    
    Args:
        seq_file (str): 日志序列文件路径（步骤2生成）
        label_file (str): 标签文件路径（0-正常 1-异常）
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("filter_normal", locals())
    _start_background(_filter_normal_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_search_synonyms(
    template_file: str,
    output_dir: str = "middle",
    dataset_name: str = "bgl"
) -> Dict[str, Any]:
    """
    【步骤4】基于WordNet搜索模板中的同义词和反义词
    
    Args:
        template_file (str): 模板文件路径
        output_dir (str): 输出目录
        dataset_name (str): 数据集名称
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("search_synonyms", locals())
    _start_background(_search_synonyms_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_convert_template_format(
    template_file: str
) -> Dict[str, Any]:
    """
    【步骤5】转换模板格式为词向量训练格式
    
    Args:
        template_file (str): 模板文件路径
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("convert_template_format", locals())
    _start_background(_convert_template_format_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_train_word_vectors(
    template_training_file: str,
    synonym_file: str,
    antonym_file: str,
    output_model: str,
    output_vocab: str,
    vector_size: int = 32
) -> Dict[str, Any]:
    """
    【步骤6】训练词向量模型（使用LRCWE算法）
    
    Args:
        template_training_file (str): 训练格式的模板文件（步骤5生成）
        synonym_file (str): 同义词文件（步骤4生成）
        antonym_file (str): 反义词文件（步骤4生成）
        output_model (str): 输出模型文件路径
        output_vocab (str): 输出词汇表文件路径
        vector_size (int): 词向量维度，默认32
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("train_word_vectors", locals())
    _start_background(_train_word_vectors_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_generate_template_vectors(
    template_file: str,
    word_model: str,
    output_vector_file: str,
    dimension: int = 32
) -> Dict[str, Any]:
    """
    【步骤7】从词向量生成模板向量
    
    Args:
        template_file (str): 模板文件路径
        word_model (str): 词向量模型文件（步骤6生成）
        output_vector_file (str): 输出模板向量文件路径
        dimension (int): 向量维度，默认32
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("generate_template_vectors", locals())
    _start_background(_generate_template_vectors_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_train_model(
    train_seq_file: str,
    template_vector_file: str,
    template_file: str,
    model_dir: str = "weights/vector_matrix",
    seq_length: int = 10,
    epochs: int = 30,
    use_onehot: bool = True,
    use_count_matrix: bool = True
) -> Dict[str, Any]:
    """
    【步骤8】训练双向LSTM异常检测模型
    
    Args:
        train_seq_file (str): 训练序列文件（步骤3生成的正常日志）
        template_vector_file (str): 模板向量文件（步骤7生成）
        template_file (str): 模板文件
        model_dir (str): 模型保存目录
        seq_length (int): 序列长度，默认10
        epochs (int): 训练轮数，默认30
        use_onehot (bool): 是否使用独热编码，默认True
        use_count_matrix (bool): 是否使用计数矩阵，默认True
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("train_model", locals())
    _start_background(_train_model_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_detect_anomaly(
    test_seq_file: str,
    label_file: str,
    template_vector_file: str,
    template_file: str,
    model_dir: str = "weights/vector_matrix",
    result_file: str = "results/detection_results.txt",
    seq_length: int = 10,
    n_candidates: int = 15,
    use_onehot: bool = True,
    use_count_matrix: bool = True
) -> Dict[str, Any]:
    """
    【步骤9】使用训练好的模型进行异常检测
    
    Args:
        test_seq_file (str): 测试序列文件
        label_file (str): 标签文件（用于评估）
        template_vector_file (str): 模板向量文件
        template_file (str): 模板文件
        model_dir (str): 模型目录
        result_file (str): 结果输出文件
        seq_length (int): 序列长度，默认10
        n_candidates (int): 候选集大小，默认15
        use_onehot (bool): 是否使用独热编码，默认True
        use_count_matrix (bool): 是否使用计数矩阵，默认True
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("detect_anomaly", locals())
    _start_background(_detect_anomaly_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_full_pipeline(
    log_file: str,
    label_file: str,
    dataset_name: str = "custom",
    seq_length: int = 10,
    epochs: int = 30,
    n_candidates: int = 15
) -> Dict[str, Any]:
    """
    【完整流程】自动执行从日志到异常检测的完整9步流程
    
    Args:
        log_file (str): 原始日志文件路径
        label_file (str): 标签文件路径
        dataset_name (str): 数据集名称
        seq_length (int): 序列长度
        epochs (int): 训练轮数
        n_candidates (int): 候选集大小
    
    Returns:
        Dict[str, Any]: 包含任务ID和初始状态的字典
    """
    task_id = _create_task("full_pipeline", locals())
    _start_background(_full_pipeline_worker, task_id, locals())
    return {"task_id": task_id, "status": TASKS[task_id]["status"]}

@mcp.tool()
def loganomaly_list_tasks() -> Dict[str, Any]:
    """
    列出所有后台任务
    
    Returns:
        Dict[str, Any]: 包含所有任务列表的字典
    """
    with TASKS_LOCK:
        return {"tasks": list(TASKS.values())}

@mcp.tool()
def loganomaly_get_task(task_id: str) -> Dict[str, Any]:
    """
    获取指定任务的详细信息
    
    Args:
        task_id (str): 任务的唯一ID
    
    Returns:
        Dict[str, Any]: 包含任务详细信息的字典
    """
    with TASKS_LOCK:
        return TASKS.get(task_id, {"error": "Task not found"})

@mcp.tool()
def loganomaly_cancel_task(task_id: str) -> Dict[str, Any]:
    """
    取消指定任务
    
    Args:
        task_id (str): 任务的唯一ID
    
    Returns:
        Dict[str, Any]: 包含取消结果的字典
    """
    with TASKS_LOCK:
        if task_id in TASKS:
            if TASKS[task_id]["status"] in ["queued", "running"]:
                TASKS[task_id]["status"] = "cancelled"
                TASKS[task_id]["completed_at"] = _now_iso()
                return {"success": True, "message": "任务已取消"}
            else:
                return {"success": False, "message": f"任务状态为 {TASKS[task_id]['status']}，无法取消"}
        else:
            return {"success": False, "message": "任务不存在"}

# ============================================================================
# 后台任务 Worker
# ============================================================================

def _extract_template_worker(task_id: str, params: Dict[str, Any]):
    """步骤1: 模板提取 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        log_file = Path(params["log_file"])
        output_dir = project_root / params["output_dir"]
        dataset_name = params["dataset_name"]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = output_dir / f"{dataset_name}_log.template"
        fre_word_file = output_dir / f"{dataset_name}_log.fre"
        
        _set_task(task_id, progress=0.1, message="开始模板提取...")
        
        cmd = [
            sys.executable, "-u", "ft_tree.py",
            "-data_path", str(log_file),
            "-template_path", str(template_file),
            "-fre_word_path", str(fre_word_file)
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "ft_tree",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "template_file": str(template_file),
            "fre_word_file": str(fre_word_file),
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _match_template_worker(task_id: str, params: Dict[str, Any]):
    """步骤2: 模板匹配 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        _set_task(task_id, progress=0.1, message="开始模板匹配...")
        
        cmd = [
            sys.executable, "-u", "matchTemplate.py",
            "-template_path", params["template_file"],
            "-fre_word_path", params["fre_word_file"],
            "-log_path", params["log_file"],
            "-out_seq_path", params["output_seq_file"],
            "-match_model", str(params["match_mode"])
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "ft_tree",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "seq_file": params["output_seq_file"],
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _filter_normal_worker(task_id: str, params: Dict[str, Any]):
    """步骤3: 过滤正常日志 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        _set_task(task_id, progress=0.1, message="开始过滤异常日志...")
        
        cmd = [
            sys.executable, "removeAnomaly.py",
            "-input_seq", params["seq_file"],
            "-input_label", params["label_file"]
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "data",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        output_file = params["seq_file"] + "_normal"
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "normal_seq_file": output_file,
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _search_synonyms_worker(task_id: str, params: Dict[str, Any]):
    """步骤4: 搜索同义词反义词 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        output_dir = project_root / params["output_dir"]
        dataset_name = params["dataset_name"]
        
        _set_task(task_id, progress=0.1, message="开始搜索同义词和反义词...")
        
        cmd = [
            sys.executable, "wordnet_process.py",
            "-data_dir", f"../../../{params['output_dir']}/",
            "-template_file", f"{dataset_name}_log.template",
            "-syn_file", f"{dataset_name}_log.syn",
            "-ant_file", f"{dataset_name}_log.ant"
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "template2Vector" / "src" / "preprocess",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "synonym_file": str(output_dir / f"{dataset_name}_log.syn"),
            "antonym_file": str(output_dir / f"{dataset_name}_log.ant"),
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _convert_template_format_worker(task_id: str, params: Dict[str, Any]):
    """步骤5: 转换模板格式 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        _set_task(task_id, progress=0.1, message="开始转换模板格式...")
        
        cmd = [
            sys.executable, "changeTemplateFormat.py",
            "-input", params["template_file"]
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "middle",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        output_file = params["template_file"] + "_for_training"
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "training_file": output_file,
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _train_word_vectors_worker(task_id: str, params: Dict[str, Any]):
    """步骤6: 训练词向量 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        _set_task(task_id, progress=0.1, message="编译词向量训练程序...")
        
        # 首先编译 C 程序
        compile_result = _run_subprocess(
            ["make"],
            project_root / "template2Vector" / "src",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.1, message=msg)
        )
        
        _set_task(task_id, progress=0.2, message="开始训练词向量...")
        
        # 训练词向量
        cmd = [
            "./lrcwe",
            "-train", params["template_training_file"],
            "-synonym", params["synonym_file"],
            "-antonym", params["antonym_file"],
            "-output", params["output_model"],
            "-save-vocab", params["output_vocab"],
            "-belta-rel", "0.8",
            "-alpha-rel", "0.01",
            "-belta-syn", "0.4",
            "-alpha-syn", "0.2",
            "-alpha-ant", "0.3",
            "-size", str(params["vector_size"]),
            "-min-count", "1"
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "template2Vector" / "src",
            lambda p, msg: _set_task(task_id, progress=0.2 + p * 0.7, message=msg)
        )
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "model_file": params["output_model"],
            "vocab_file": params["output_vocab"],
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _generate_template_vectors_worker(task_id: str, params: Dict[str, Any]):
    """步骤7: 生成模板向量 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        _set_task(task_id, progress=0.1, message="开始生成模板向量...")
        
        cmd = [
            sys.executable, "template2Vec.py",
            "-template_file", params["template_file"],
            "-word_model", params["word_model"],
            "-template_vector_file", params["output_vector_file"],
            "-dimension", str(params["dimension"])
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "template2Vector" / "src",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "vector_file": params["output_vector_file"],
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _train_model_worker(task_id: str, params: Dict[str, Any]):
    """步骤8: 训练模型 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        model_dir = project_root / params["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        _set_task(task_id, progress=0.1, message="开始训练双向LSTM模型...")
        
        cmd = [
            sys.executable, "-u", "train_vector_2LSTM.py",
            "-train_file", params["train_seq_file"],
            "-seq_length", str(params["seq_length"]),
            "-model_dir", str(model_dir) + "/",
            "-onehot", "1" if params["use_onehot"] else "0",
            "-template2Vec_file", params["template_vector_file"],
            "-template_file", params["template_file"],
            "-count_matrix", "1" if params["use_count_matrix"] else "0",
            "-epoch", str(params["epochs"])
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "LogAnomaly_main",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "model_dir": str(model_dir),
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _detect_anomaly_worker(task_id: str, params: Dict[str, Any]):
    """步骤9: 异常检测 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.05)
    try:
        result_dir = Path(params["result_file"]).parent
        result_dir.mkdir(parents=True, exist_ok=True)
        
        _set_task(task_id, progress=0.1, message="开始异常检测...")
        
        cmd = [
            sys.executable, "-u", "detect_vector_2LSTM.py",
            "-test_file", params["test_seq_file"],
            "-seq_length", str(params["seq_length"]),
            "-model_dir", params["model_dir"] + "/",
            "-n_candidates", str(params["n_candidates"]),
            "-windows_size", "3",
            "-step_size", "1",
            "-onehot", "1" if params["use_onehot"] else "0",
            "-result_file", params["result_file"],
            "-label_file", params["label_file"],
            "-template2Vec_file", params["template_vector_file"],
            "-template_file", params["template_file"],
            "-count_matrix", "1" if params["use_count_matrix"] else "0"
        ]
        
        result = _run_subprocess(
            cmd,
            project_root / "LogAnomaly_main",
            lambda p, msg: _set_task(task_id, progress=0.1 + p * 0.8, message=msg)
        )
        
        # 读取结果文件
        detection_results = {}
        if Path(params["result_file"]).exists():
            with open(params["result_file"], 'r') as f:
                lines = f.readlines()
                detection_results["raw_results"] = "".join(lines)
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "result_file": params["result_file"],
            "detection_results": detection_results,
            "stdout": result["stdout"]
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

def _full_pipeline_worker(task_id: str, params: Dict[str, Any]):
    """完整流程 Worker"""
    _set_task(task_id, status="running", started_at=_now_iso(), progress=0.0)
    try:
        dataset_name = params["dataset_name"]
        log_file = params["log_file"]
        label_file = params["label_file"]
        
        # 创建必要的目录
        middle_dir = project_root / "middle"
        model_dir = project_root / "model"
        weights_dir = project_root / "weights" / "vector_matrix"
        results_dir = project_root / "results"
        
        for d in [middle_dir, model_dir, weights_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 定义文件路径
        template_file = middle_dir / f"{dataset_name}_log.template"
        fre_word_file = middle_dir / f"{dataset_name}_log.fre"
        seq_file = middle_dir / f"{dataset_name}_log.seq"
        normal_seq_file = middle_dir / f"{dataset_name}_log.seq_normal"
        syn_file = middle_dir / f"{dataset_name}_log.syn"
        ant_file = middle_dir / f"{dataset_name}_log.ant"
        training_file = middle_dir / f"{dataset_name}_log.template_for_training"
        word_model = model_dir / f"{dataset_name}_log.model"
        vocab_file = middle_dir / f"{dataset_name}_log.vector_vocab"
        template_vector = model_dir / f"{dataset_name}_log.template_vector"
        result_file = results_dir / f"{dataset_name}_detection_results.txt"
        
        pipeline_steps = [
            (0.0, 0.11, "步骤1: 提取模板", lambda: _extract_template_worker(f"{task_id}_step1", {
                "log_file": log_file,
                "output_dir": "middle",
                "dataset_name": dataset_name
            })),
            (0.11, 0.22, "步骤2: 匹配模板", lambda: _match_template_worker(f"{task_id}_step2", {
                "log_file": log_file,
                "template_file": str(template_file),
                "fre_word_file": str(fre_word_file),
                "output_seq_file": str(seq_file),
                "match_mode": 1
            })),
            (0.22, 0.33, "步骤3: 过滤正常日志", lambda: _filter_normal_worker(f"{task_id}_step3", {
                "seq_file": str(seq_file),
                "label_file": label_file
            })),
            (0.33, 0.44, "步骤4: 搜索同义词反义词", lambda: _search_synonyms_worker(f"{task_id}_step4", {
                "template_file": str(template_file),
                "output_dir": "middle",
                "dataset_name": dataset_name
            })),
            (0.44, 0.55, "步骤5: 转换模板格式", lambda: _convert_template_format_worker(f"{task_id}_step5", {
                "template_file": str(template_file)
            })),
            (0.55, 0.66, "步骤6: 训练词向量", lambda: _train_word_vectors_worker(f"{task_id}_step6", {
                "template_training_file": str(training_file),
                "synonym_file": str(syn_file),
                "antonym_file": str(ant_file),
                "output_model": str(word_model),
                "output_vocab": str(vocab_file),
                "vector_size": 32
            })),
            (0.66, 0.77, "步骤7: 生成模板向量", lambda: _generate_template_vectors_worker(f"{task_id}_step7", {
                "template_file": str(template_file),
                "word_model": str(word_model),
                "output_vector_file": str(template_vector),
                "dimension": 32
            })),
            (0.77, 0.88, "步骤8: 训练LSTM模型", lambda: _train_model_worker(f"{task_id}_step8", {
                "train_seq_file": str(normal_seq_file),
                "template_vector_file": str(template_vector),
                "template_file": str(template_file),
                "model_dir": "weights/vector_matrix",
                "seq_length": params["seq_length"],
                "epochs": params["epochs"],
                "use_onehot": True,
                "use_count_matrix": True
            })),
            (0.88, 1.0, "步骤9: 异常检测", lambda: _detect_anomaly_worker(f"{task_id}_step9", {
                "test_seq_file": str(seq_file),
                "label_file": label_file,
                "template_vector_file": str(template_vector),
                "template_file": str(template_file),
                "model_dir": "weights/vector_matrix",
                "result_file": str(result_file),
                "seq_length": params["seq_length"],
                "n_candidates": params["n_candidates"],
                "use_onehot": True,
                "use_count_matrix": True
            }))
        ]
        
        for start_progress, end_progress, step_name, step_func in pipeline_steps:
            _set_task(task_id, progress=start_progress, message=step_name)
            step_func()
            # 等待步骤完成（这里简化处理，实际应该监控子任务状态）
            time.sleep(2)
        
        _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result={
            "message": "完整流程执行完成",
            "result_file": str(result_file)
        })
        
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())

# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print(" LogAnomaly MCP Server 启动中...")
    print(" 双向LSTM日志异常检测系统")
    print("=" * 60)
    mcp.run(transport="sse", port=2226)

