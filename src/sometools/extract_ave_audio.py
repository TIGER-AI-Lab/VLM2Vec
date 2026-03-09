#!/usr/bin/env python3
"""
从AVE数据集的视频文件中提取音频片段，用于audio-to-video检索任务
"""

import os
import subprocess
import argparse
from typing import List, Dict, Any


def parse_ave_split(split_file: str) -> List[Dict[str, Any]]:
    """
    解析AVE split文件
    """
    samples = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Category"):
                continue
            parts = line.split("&")
            if len(parts) < 5:
                continue
            category, video_id, quality, start, end = parts[:5]
            clip_id = f"{video_id}_{int(float(start))}_{int(float(end))}"
            samples.append({
                "category": category,
                "video_id": video_id,
                "quality": quality,
                "start": float(start),
                "end": float(end),
                "clip_id": clip_id,
            })
    return samples


def extract_audio_segment(video_path: str, audio_path: str, start_time: float, duration: float) -> bool:
    """
    使用ffmpeg从视频中提取音频片段
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,  # 输入视频文件
        "-ss", str(start_time),  # 开始时间
        "-t", str(duration),  # 持续时间
        "-vn",  # 不处理视频
        "-acodec", "pcm_s16le",  # 音频编码为WAV
        "-ar", "16000",  # 采样率16kHz (适合语音任务)
        "-ac", "1",  # 单声道
        "-y",  # 覆盖输出文件
        audio_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Timeout extracting audio from {video_path}")
        return False
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False


def main():
    print("Starting main function")
    parser = argparse.ArgumentParser(description="Extract audio segments from AVE dataset videos")
    parser.add_argument("--data_path", type=str,
                       default="/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/ave/AVE",
                       help="AVE数据集根目录")
    parser.add_argument("--split_file", type=str, default="testSet.txt",
                       help="分割文件 (testSet.txt, trainSet.txt, valSet.txt)")
    parser.add_argument("--video_dir", type=str, default="AVE_Dataset/AVE",
                       help="视频目录名")
    parser.add_argument("--audio_dir", type=str, default="AVE_Dataset/audios",
                       help="输出音频目录名")
    parser.add_argument("--force", action="store_true",
                       help="强制重新提取已存在的音频文件")

    print("Parsing arguments")
    args = parser.parse_args()
    print("Arguments parsed")

    # 构建路径
    split_path = os.path.join(args.data_path, "AVE_Dataset", args.split_file)
    video_path = os.path.join(args.data_path, args.video_dir)
    audio_path = os.path.join(args.data_path, args.audio_dir)

    # 创建音频目录
    os.makedirs(audio_path, exist_ok=True)

    # 解析分割文件
    print(f"解析分割文件: {split_path}")
    samples = parse_ave_split(split_path)
    print(f"找到 {len(samples)} 个音频片段")

    # 统计信息
    success_count = 0
    skip_count = 0
    fail_count = 0

    # 处理每个样本
    print(f"开始处理 {len(samples)} 个音频片段...")
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(samples)} 个片段")
        # 根据testSet.txt，视频文件名就是video_id.mp4
        video_file = f"{sample['video_id']}.mp4"

        video_abs_path = os.path.join(video_path, video_file)

        # 检查视频文件是否存在
        if not os.path.isfile(video_abs_path):
            print(f"警告: 视频文件不存在 {video_abs_path}")
            fail_count += 1
            continue

        # 构建音频文件路径
        audio_file = f"{sample['video_id']}_{sample['clip_id']}.wav"
        audio_abs_path = os.path.join(audio_path, audio_file)

        # 检查音频文件是否已存在
        if os.path.isfile(audio_abs_path) and not args.force:
            skip_count += 1
            continue

        # 计算持续时间
        duration = sample['end'] - sample['start']
        if duration <= 0:
            print(f"警告: 无效的持续时间 {duration} 对于 {sample['clip_id']}")
            fail_count += 1
            continue

        # 提取音频
        if extract_audio_segment(video_abs_path, audio_abs_path, sample['start'], duration):
            success_count += 1
        else:
            fail_count += 1
            if os.path.exists(audio_abs_path):
                os.remove(audio_abs_path)  # 删除失败的文件

    # 输出统计信息
    print("\n处理完成:")
    print(f"  成功提取: {success_count}")
    print(f"  跳过已存在: {skip_count}")
    print(f"  提取失败: {fail_count}")
    print(f"  总计处理: {len(samples)}")


if __name__ == "__main__":
    main()
