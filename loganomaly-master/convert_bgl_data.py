#!/usr/bin/env python3
"""
BGL 数据格式转换脚本
将 BGL_2k.log_structured.csv 转换为 LogAnomaly 所需的格式
"""

import pandas as pd
import os
from pathlib import Path

def convert_bgl_data():
    """转换 BGL CSV 数据为 LogAnomaly 格式"""
    
    # 输入和输出文件路径
    input_file = Path("data/BGL_2k.log_structured.csv")
    output_log_file = Path("data/bgl.log")
    output_label_file = Path("data/bgl.label")
    
    print("🔄 开始转换 BGL 数据格式...")
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出日志文件: {output_log_file}")
    print(f"📁 输出标签文件: {output_label_file}")
    print()
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return False
    
    try:
        # 读取 CSV 文件
        print("📖 读取 CSV 文件...")
        df = pd.read_csv(input_file)
        
        print(f"📊 数据概览:")
        print(f"   - 总行数: {len(df)}")
        print(f"   - 列数: {len(df.columns)}")
        print(f"   - 列名: {list(df.columns)}")
        
        # 检查标签分布
        print(f"\n📈 标签分布:")
        label_counts = df['Label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - '{label}': {count} 条 ({percentage:.2f}%)")
        
        # 构建日志内容
        print(f"\n🔧 构建日志内容...")
        log_contents = []
        labels = []
        
        for _, row in df.iterrows():
            # 构建日志行格式: 时间戳 节点 级别 内容
            timestamp = row['Timestamp']
            node = row['Node']
            level = row['Level']
            content = row['Content']
            
            # 清理内容中的换行符和多余空格
            content = str(content).replace('\n', ' ').replace('\r', ' ').strip()
            
            # 构建日志行
            log_line = f"{timestamp} {node} {level}: {content}"
            log_contents.append(log_line)
            
            # 转换标签: '-' 表示正常(0), 其他表示异常(1)
            if row['Label'] == '-':
                labels.append('0')
            else:
                labels.append('1')
        
        # 保存日志文件
        print(f"💾 保存日志文件: {output_log_file}")
        with open(output_log_file, 'w', encoding='utf-8') as f:
            for log_line in log_contents:
                f.write(log_line + '\n')
        
        # 保存标签文件
        print(f"💾 保存标签文件: {output_label_file}")
        with open(output_label_file, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(label + '\n')
        
        # 验证转换结果
        print(f"\n✅ 转换完成!")
        print(f"   - 日志文件行数: {len(log_contents)}")
        print(f"   - 标签文件行数: {len(labels)}")
        
        # 统计异常情况
        normal_count = labels.count('0')
        anomaly_count = labels.count('1')
        anomaly_rate = (anomaly_count / len(labels)) * 100
        
        print(f"\n📊 转换后统计:")
        print(f"   - 正常日志: {normal_count} 条")
        print(f"   - 异常日志: {anomaly_count} 条")
        print(f"   - 异常率: {anomaly_rate:.2f}%")
        
        # 显示前几行示例
        print(f"\n📋 日志文件前5行示例:")
        for i in range(min(5, len(log_contents))):
            print(f"   {i+1}. {log_contents[i]}")
        
        print(f"\n📋 标签文件前10个标签:")
        print(f"   {' '.join(labels[:10])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = convert_bgl_data()
    if success:
        print(f"\n🎉 BGL 数据转换成功!")
        print(f"现在可以使用以下命令进行异常检测:")
        print(f"  '帮我对 data/bgl.log 进行完整的异常检测，标签文件是 data/bgl.label'")
    else:
        print(f"\n💥 BGL 数据转换失败!")
