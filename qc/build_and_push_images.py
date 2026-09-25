#!/usr/bin/env python3
"""Build and push Docker images from JSONL instance records. Parse inputs, build PR images, authenticate to the configured registry, then retag and push."""

import os
os.environ.setdefault("STEM_DEVNULL", os.devnull)
import json
import subprocess
import logging
import sys
import glob
from pathlib import Path
from typing import Dict, Set, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

# Configuration: input data directories.

DATA_DIR = "./need_build/ruby"  # Directory containing JSONL files.

# Container registry configuration.
REGISTRY = os.getenv("DOCKER_REGISTRY", "registry.example.com")
NAMESPACE = os.getenv("DOCKER_NAMESPACE", "anonymous")
USERNAME = os.getenv("DOCKER_USERNAME", "")
PASSWORD = os.getenv("DOCKER_PASSWORD", "")

# Build configuration.
WORKDIR = "./workdir"  # Working directory.
REPO_DIR = "./repos"  # Repository directory.
LOG_DIR = "./logs"  # Log directory.

# Logging configuration.
LOG_FILE = "build_and_push_images.log"

# Build parameters.
MAX_WORKERS_BUILD = 5  # Maximum concurrent image-build workers.
FORCE_BUILD = False  # Whether to force rebuilding existing images.

# Concurrency settings.
MAX_WORKERS_PUSH = 10  # Maximum image-push workers and concurrent JSONL processing jobs.
MAX_WORKERS_JSONL = 3  

# Cleanup settings.
AUTO_CLEANUP = True  # Remove local images after successful pushes to release disk space.


# ===========================================


@dataclass
class ImageInfo:
    """Container image metadata."""
    instance_id: str
    org: str
    repo: str
    pr_number: int
    original_image: str  # Source image name in mswebench format.
    target_image: str  # Destination registry image name.


# Log to both the console and a file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


class ThreadSafeCounter:
    """Thread-safe progress counter."""

    def __init__(self):
        self._lock = threading.Lock()
        self.success = 0
        self.failed = 0

    def increment_success(self):
        with self._lock:
            self.success += 1

    def increment_failed(self):
        with self._lock:
            self.failed += 1

    def get_counts(self):
        with self._lock:
            return self.success, self.failed


def run_command(command: str, hide_output: bool = False) -> tuple[bool, str]:
    """Execute and log a command. hide_output suppresses sensitive command output. Return a success flag and captured output."""
    if not hide_output:
        logging.info(f"执行命令: {command}")
    else:
        logging.info(f"执行命令: [命令已隐藏]")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        logging.error(f"命令执行失败！错误信息:\n{stderr.strip()}")
        return False, stderr.strip()

    return True, stdout.strip()


def login_registry() -> bool:
    """Authenticate to the container registry."""
    logging.info(f"--- 尝试登录注册表: {REGISTRY} ---")
    cmd = f"docker login --username {USERNAME} --password {PASSWORD} {REGISTRY}"
    success, output = run_command(cmd, hide_output=True)

    if success:
        logging.info("登录成功！")
        return True
    else:
        logging.error(f"登录失败: {output}")
        return False


def parse_jsonl_files(data_dir: str) -> Dict[str, ImageInfo]:
    """Parse JSONL files in data_dir and return an instance-ID-to-ImageInfo mapping."""
    logging.info(f"开始扫描目录: {data_dir}")

    if not os.path.exists(data_dir):
        logging.error(f"路径不存在: {data_dir}")
        return {}

    image_map: Dict[str, ImageInfo] = {}
    jsonl_files = glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)

    if not jsonl_files:
        logging.warning(f"在 {data_dir} 中未找到任何 .jsonl 文件")
        return {}

    logging.info(f"找到 {len(jsonl_files)} 个 jsonl 文件")

    for file_path in jsonl_files:
        logging.info(f"正在解析文件: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        instance_id = data.get("instance_id")

                        if not instance_id:
                            logging.warning(f"文件 {file_path} 第 {line_num} 行缺少 instance_id，跳过")
                            continue

                        # Prefer explicit org, repo, and number fields from JSONL.
                        org = data.get("org")
                        repo = data.get("repo")
                        pr_number = data.get("number")

                        # Otherwise parse the instance ID.
                        if not (org and repo and pr_number):
                            # Support org__repo-pr-number and org__repo-number instance IDs.


                            if "-pr-" in instance_id:
                                # Parse the format containing -pr-.
                                parts = instance_id.split("-pr-")
                                if len(parts) != 2:
                                    logging.warning(f"无法解析 instance_id: {instance_id}")
                                    continue
                                repo_part = parts[0]
                                pr_number = int(parts[1])
                            else:
                                # Otherwise split at the final hyphen.
                                
                                last_dash = instance_id.rfind("-")
                                if last_dash == -1:
                                    logging.warning(f"无法解析 instance_id: {instance_id}")
                                    continue
                                repo_part = instance_id[:last_dash]
                                try:
                                    pr_number = int(instance_id[last_dash + 1:])
                                except ValueError:
                                    logging.warning(f"无法解析 PR 号: {instance_id}")
                                    continue

                            # Parse organization and repository names.
                            org_repo = repo_part.split("__")
                            if len(org_repo) != 2:
                                logging.warning(f"无法解析 org/repo: {repo_part}")
                                continue

                            org = org_repo[0]
                            repo = org_repo[1]

                        # Construct the lowercase source image name as mswebench/{org}_m_{repo}:pr-{number}.
                        
                        
                        original_image = f"mswebench/{org}_m_{repo}:pr-{pr_number}".lower()

                        # Construct the lowercase destination registry image name.
                        
                        
                        target_image = f"{REGISTRY}/{NAMESPACE}/multi-swe-rl.test.{instance_id}".lower()

                        image_info = ImageInfo(
                            instance_id=instance_id,
                            org=org,
                            repo=repo,
                            pr_number=pr_number,
                            original_image=original_image,
                            target_image=target_image
                        )

                        image_map[instance_id] = image_info
                        logging.debug(f"解析成功: {instance_id} -> {original_image}")

                    except json.JSONDecodeError:
                        logging.error(f"文件 {file_path} 第 {line_num} 行 JSON 格式错误")
                    except Exception as e:
                        logging.error(f"文件 {file_path} 第 {line_num} 行解析出错: {e}")

        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")

    logging.info(f"共解析出 {len(image_map)} 个有效的镜像信息")
    return image_map


def build_images_with_build_dataset(jsonl_file: str) -> bool:
    """Build images from jsonl_file using build_dataset.py; return whether the build succeeded."""
    logging.info(f"=" * 60)
    logging.info(f"开始构建镜像: {jsonl_file}")
    logging.info(f"=" * 60)

    # Create required directories.
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(REPO_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Construct the build command.
    cmd = (
        f"python3 -m multi_swe_bench.harness.build_dataset "
        f"--mode image "
        f"--workdir {WORKDIR} "
        f"--raw_dataset_files {jsonl_file} "
        f"--repo_dir {REPO_DIR} "
        f"--log_dir {LOG_DIR} "
        f"--max_workers_build_image {MAX_WORKERS_BUILD} "
        f"--force_build {str(FORCE_BUILD).lower()}"
    )

    success, output = run_command(cmd)

    if success:
        logging.info(f"镜像构建成功: {jsonl_file}")
        return True
    else:
        logging.error(f"镜像构建失败: {jsonl_file}")
        return False


def check_image_exists(image_name: str) -> bool:
    """Check whether an image exists locally."""
    cmd = f"docker inspect {image_name} > ${{STEM_DEVNULL}} 2>&1"
    process = subprocess.run(cmd, shell=True)
    return process.returncode == 0


def cleanup_local_image(image_info: ImageInfo) -> bool:
    """Remove both local image tags for image_info; return whether cleanup succeeded."""
    logging.info(f"[清理镜像] {image_info.instance_id}")

    success = True

    # Remove the destination-tagged local image.
    if check_image_exists(image_info.target_image):
        cmd = f"docker rmi {image_info.target_image}"
        result, _ = run_command(cmd)
        if result:
            logging.info(f"  ✓ 已删除目标镜像: {image_info.target_image}")
        else:
            logging.warning(f"  ✗ 删除目标镜像失败: {image_info.target_image}")
            success = False

    # Remove the source-tagged local image.
    if check_image_exists(image_info.original_image):
        cmd = f"docker rmi {image_info.original_image}"
        result, _ = run_command(cmd)
        if result:
            logging.info(f"  ✓ 已删除原始镜像: {image_info.original_image}")
        else:
            logging.warning(f"  ✗ 删除原始镜像失败: {image_info.original_image}")
            success = False

    return success


def tag_and_push_image(image_info: ImageInfo, counter: ThreadSafeCounter = None, total: int = 0) -> bool:
    """Tag and push image_info, optionally updating a shared counter and total; return success."""
    logging.info("-" * 60)
    logging.info(f"[并发推送] 处理镜像: {image_info.instance_id}")
    logging.info(f"  原始镜像: {image_info.original_image}")
    logging.info(f"  目标镜像: {image_info.target_image}")

    try:
        # Check that the source image exists.
        if not check_image_exists(image_info.original_image):
            logging.error(f"[失败] 原始镜像不存在: {image_info.original_image}")
            if counter:
                counter.increment_failed()
            return False

        # Get the image ID.
        success, image_id = run_command(f"docker inspect -f '{{{{.Id}}}}' {image_info.original_image}")
        if not success:
            logging.error(f"[失败] 无法获取镜像ID: {image_info.original_image}")
            if counter:
                counter.increment_failed()
            return False

        logging.info(f"  镜像ID: {image_id}")

        # Apply the destination tag.
        logging.info(f"[打标签] {image_info.instance_id}")
        success, _ = run_command(f"docker tag {image_id} {image_info.target_image}")
        if not success:
            logging.error(f"[失败] 打标签失败: {image_info.instance_id}")
            if counter:
                counter.increment_failed()
            return False

        # Push the image.
        logging.info(f"[推送中] {image_info.instance_id} -> {REGISTRY}")
        success, _ = run_command(f"docker push {image_info.target_image}")
        if not success:
            logging.error(f"[失败] 推送失败: {image_info.instance_id}")
            if counter:
                counter.increment_failed()
            return False

        logging.info(f"✓ [成功] 推送完成: {image_info.target_image}")

        # Release local disk space after a successful push.
        if AUTO_CLEANUP:
            cleanup_local_image(image_info)

        if counter:
            counter.increment_success()
            success_count, failed_count = counter.get_counts()
            remaining = total - success_count - failed_count
            logging.info(f"[进度] 成功: {success_count}, 失败: {failed_count}, 剩余: {remaining}")

        return True

    except Exception as e:
        logging.error(f"[异常] 处理镜像 {image_info.instance_id} 时发生异常: {e}", exc_info=True)
        if counter:
            counter.increment_failed()
        return False


def group_by_jsonl(image_map: Dict[str, ImageInfo]) -> Dict[str, list[ImageInfo]]:
    """Group image records by their source JSONL path."""
    groups: Dict[str, list[ImageInfo]] = {}

    for image_info in image_map.values():
        # Locate the JSONL input from organization and repository metadata under the language directory.
        
        
        pattern = f"{DATA_DIR}/**/{image_info.org}__{image_info.repo}_dataset.jsonl"
        matches = glob.glob(pattern, recursive=True)

        if matches:
            jsonl_file = matches[0]
            if jsonl_file not in groups:
                groups[jsonl_file] = []
            groups[jsonl_file].append(image_info)

    return groups


def push_images_parallel(images: List[ImageInfo], counter: ThreadSafeCounter, total: int) -> tuple[int, int]:
    """Push images concurrently and return success and failure counts."""
    logging.info(f"[并行推送] 开始并行推送 {len(images)} 个镜像，并发数: {MAX_WORKERS_PUSH}")

    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PUSH) as executor:
        # Submit all jobs.
        future_to_image = {
            executor.submit(tag_and_push_image, img, counter, total): img
            for img in images
        }

        # Wait for completion.
        for future in as_completed(future_to_image):
            image_info = future_to_image[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logging.error(f"[异常] 推送镜像 {image_info.instance_id} 时发生异常: {e}", exc_info=True)
                failed_count += 1
                counter.increment_failed()

    logging.info(f"[并行推送完成] 成功: {success_count}, 失败: {failed_count}")
    return success_count, failed_count


def process_single_jsonl(jsonl_file: str, images: List[ImageInfo], counter: ThreadSafeCounter, total: int) -> tuple[
    int, int]:
    """Process one JSONL file by building and immediately pushing each image; return success and failure counts."""
    logging.info(f"\n{'=' * 80}")
    logging.info(f"[开始处理] 文件: {jsonl_file}")
    logging.info(f"[镜像数量] {len(images)} 个镜像")
    logging.info(f"[处理策略] 逐个构建并立即推送（边构建边推送，节省磁盘空间）")
    logging.info(f"{'=' * 80}\n")

    success_count = 0
    failed_count = 0

    # Process each image through build, push, and cleanup.
    for idx, image_info in enumerate(images, 1):
        logging.info(f"\n[{idx}/{len(images)}] 处理镜像: {image_info.instance_id}")

        # Create a temporary JSONL input for one image.
        temp_jsonl = f"{WORKDIR}/temp_{image_info.instance_id}.jsonl"
        os.makedirs(WORKDIR, exist_ok=True)

        try:
            # Extract the matching record from the original input.
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('instance_id') == image_info.instance_id:
                        with open(temp_jsonl, 'w', encoding='utf-8') as temp_f:
                            temp_f.write(line)
                        break

            # Build one image.
            if not build_images_with_build_dataset(temp_jsonl):
                logging.error(f"[构建失败] {image_info.instance_id}")
                counter.increment_failed()
                failed_count += 1
                continue

            # Push it immediately.
            if tag_and_push_image(image_info, counter, total):
                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            logging.error(f"[异常] 处理镜像 {image_info.instance_id} 时发生异常: {e}", exc_info=True)
            counter.increment_failed()
            failed_count += 1

        finally:
            # Remove the temporary input.
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)

    logging.info(f"\n[文件处理完成] {jsonl_file}")
    logging.info(f"  成功: {success_count}, 失败: {failed_count}")

    return success_count, failed_count


def main():
    """Command-line entry point."""
    logging.info("=" * 80)
    logging.info("开始批量构建和推送Docker镜像")
    logging.info(f"[并发配置] 推送并发数: {MAX_WORKERS_PUSH}, JSONL并发数: {MAX_WORKERS_JSONL}")
    logging.info(f"[清理策略] 自动清理: {'开启' if AUTO_CLEANUP else '关闭'} (推送成功后立即删除本地镜像)")
    logging.info(f"[处理策略] 边构建边推送（逐个处理，节省磁盘空间）")
    logging.info("=" * 80)

    # Authenticate to the registry.
    if not login_registry():
        logging.error("登录失败，程序退出")
        sys.exit(1)

    # Parse JSONL inputs.
    image_map = parse_jsonl_files(DATA_DIR)
    if not image_map:
        logging.error("未找到任何有效的镜像信息，程序退出")
        sys.exit(1)

    # Group records by input file.
    groups = group_by_jsonl(image_map)
    logging.info(f"[任务统计] 共需要处理 {len(groups)} 个 jsonl 文件，总计 {len(image_map)} 个镜像")

    # Create a thread-safe counter.
    counter = ThreadSafeCounter()
    total_images = len(image_map)

    # Process JSONL files concurrently.
    logging.info(f"\n{'=' * 80}")
    logging.info(f"[开始并行处理] 使用 {MAX_WORKERS_JSONL} 个进程并行处理 jsonl 文件")
    logging.info(f"{'=' * 80}\n")

    # Use threads to share counters and logging state.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_JSONL) as executor:
        # Submit one job per JSONL input.
        future_to_jsonl = {
            executor.submit(process_single_jsonl, jsonl_file, images, counter, total_images): jsonl_file
            for jsonl_file, images in groups.items()
        }

        # Wait for all jobs.
        completed = 0
        for future in as_completed(future_to_jsonl):
            jsonl_file = future_to_jsonl[future]
            completed += 1
            try:
                success, failed = future.result()
                logging.info(f"[JSONL完成] ({completed}/{len(groups)}) {jsonl_file}")
            except KeyboardInterrupt:
                logging.info("\n用户中断，正在等待当前任务完成...")
                executor.shutdown(wait=True, cancel_futures=True)
                break
            except Exception as e:
                logging.error(f"[JSONL异常] 处理文件 {jsonl_file} 时发生异常: {e}", exc_info=True)

    # Report aggregate statistics.
    final_success, final_failed = counter.get_counts()

    logging.info("\n" + "=" * 80)
    logging.info("处理完成！")
    logging.info(f"总计: {total_images} 个镜像")
    logging.info(f"成功: {final_success} 个")
    logging.info(f"失败: {final_failed} 个")
    logging.info(f"成功率: {final_success / total_images * 100:.2f}%")
    logging.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logging.error(f"程序异常退出: {e}", exc_info=True)
        sys.exit(1)
