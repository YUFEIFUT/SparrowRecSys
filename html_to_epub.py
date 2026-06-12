#!/usr/bin/env python3
"""将docs文件夹下的HTML文件合并为一个EPUB电子书，按章节组织。
多线程下载在线图片，处理本地和base64图片。
"""

import os
import re
import base64
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from ebooklib import epub
import urllib.request

DOCS_DIR = Path(__file__).parent / "docs"
OUTPUT_FILE = Path(__file__).parent / "深度学习推荐系统实战.epub"

CHAPTER_DIRS = [
    "1、开篇词", "2、基础架构", "3、特征工程", "4、线上服务",
    "5、推荐模型", "6、模型评估", "7、前沿拓展", "8、结束",
]

BOOK_TITLE = "深度学习推荐系统实战"
BOOK_AUTHOR = "王喆"

global_img_counter = 0
image_cache = {}  # cache_key -> (img_name, img_data, mime)
download_lock = __import__("threading").Lock()


def get_chapter_order(filename):
    match = re.search(r"(\d+)[\-_\s]", filename)
    if match:
        return int(match.group(1))
    for kw, val in [("特别加餐", 99), ("加餐", 99), ("答疑", 98), ("期末", 97), ("国庆", 96), ("模型实战准备", 50)]:
        if kw in filename:
            return val
    return 100


def download_one_image(url):
    """下载单张图片，返回 (url, data, mime, ext) 或 (url, None, None, None)。"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "png" in ct:
                mime, ext = "image/png", "png"
            elif "gif" in ct:
                mime, ext = "image/gif", "gif"
            elif "svg" in ct:
                mime, ext = "image/svg+xml", "svg"
            elif "webp" in ct:
                mime, ext = "image/webp", "webp"
            else:
                mime, ext = "image/jpeg", "jpg"
            return url, data, mime, ext
    except Exception as e:
        return url, None, None, None


def batch_download_images(urls):
    """多线程批量下载图片，返回 {url: (data, mime, ext)}。"""
    results = {}
    unique_urls = list(set(urls))
    print(f"  下载 {len(unique_urls)} 张在线图片...")

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(download_one_image, u): u for u in unique_urls}
        done = 0
        for fut in as_completed(futures):
            url, data, mime, ext = fut.result()
            done += 1
            if data:
                results[url] = (data, mime, ext)
            if done % 50 == 0 or done == len(unique_urls):
                print(f"    进度: {done}/{len(unique_urls)}")

    print(f"  下载完成: {len(results)}/{len(unique_urls)} 成功")
    return results


def process_all_html():
    """预处理所有HTML文件，收集图片URL并下载。"""
    global global_img_counter

    all_html_data = []  # [(chapter_dir_name, html_file)]

    for chapter_dir_name in CHAPTER_DIRS:
        chapter_path = DOCS_DIR / chapter_dir_name
        if not chapter_path.exists():
            continue
        for html_file in sorted(chapter_path.glob("*.html"), key=lambda f: get_chapter_order(f.name)):
            all_html_data.append((chapter_dir_name, html_file))

    # 第一遍：收集所有在线图片URL
    online_urls = []
    local_images = []  # [(html_path, src)]

    for chapter_dir_name, html_file in all_html_data:
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src.startswith("http://") or src.startswith("https://"):
                online_urls.append(src)
            elif src.startswith("./") or src.startswith("../"):
                img_path = (html_file.parent / src).resolve()
                if img_path.exists():
                    local_images.append((str(html_path), str(img_path)))

    # 批量下载在线图片
    downloaded = batch_download_images(online_urls)

    # 第二遍：构建EPUB内容
    return all_html_data, downloaded, local_images


def extract_content(html_path, downloaded_images):
    """提取文章内容，处理所有图片。"""
    global global_img_counter

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else Path(html_path).stem

    body = soup.find("body")
    if not body:
        return title, "<p>内容为空</p>", []

    content_div = None
    for div in body.find_all("div", recursive=True):
        if div.find("h1"):
            content_div = div
            break
    if not content_div:
        content_div = body

    for tag in content_div.find_all(["audio", "button", "textarea", "script", "style", "noscript"]):
        tag.decompose()

    for div in content_div.find_all("div"):
        text = div.get_text(strip=True)
        if any(kw in text for kw in ("立即订阅", "提交留言", "Ctrl + Enter", "该试读文章来自", "精选留言")):
            div.decompose()

    images = []
    html_dir = Path(html_path).parent

    for img in content_div.find_all("img"):
        src = img.get("src", "")
        if not src or src.startswith("javascript:"):
            continue

        # Base64
        if src.startswith("data:"):
            match = re.match(r"data:(image/[\w+]+);base64,(.*)", src, re.DOTALL)
            if match:
                mime = match.group(1)
                data = base64.b64decode(match.group(2))
                ext = mime.split("/")[-1].replace("+xml", "")
                with download_lock:
                    global_img_counter += 1
                    img_name = f"img_{global_img_counter}.{ext}"
                img["src"] = f"images/{img_name}"
                images.append((img_name, data, mime))
            continue

        # 在线URL
        if src.startswith("http://") or src.startswith("https://"):
            if src in downloaded_images:
                data, mime, ext = downloaded_images[src]
                with download_lock:
                    global_img_counter += 1
                    img_name = f"img_{global_img_counter}.{ext}"
                img["src"] = f"images/{img_name}"
                images.append((img_name, data, mime))
            else:
                img.decompose()  # 下载失败则移除
            continue

        # 本地文件
        img_path = (html_dir / src).resolve()
        if img_path.exists() and img_path.is_file():
            ext = img_path.suffix.lstrip(".")
            if ext not in ("jpg", "jpeg", "png", "gif", "svg", "webp", "bmp"):
                ext = "jpg"
            mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
            with open(img_path, "rb") as f:
                data = f.read()
            with download_lock:
                global_img_counter += 1
                img_name = f"img_{global_img_counter}.{ext}"
            img["src"] = f"images/{img_name}"
            images.append((img_name, data, mime))

    h1 = content_div.find("h1")
    article_title = h1.get_text(strip=True) if h1 else title

    body_content = "".join(str(child) for child in content_div.children if hasattr(child, "name") and child.name)

    styled_html = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>{article_title}</title>
<meta charset="utf-8"/>
<style>
body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.8; color: #333; padding: 0.5em; }}
h1 {{ font-size: 1.5em; margin-bottom: 0.5em; }}
h2 {{ font-size: 1.2em; margin-top: 1.2em; color: #2c3e50; }}
p {{ margin: 0.6em 0; text-align: justify; }}
strong {{ color: #c0392b; }}
img {{ max-width: 100%; height: auto; margin: 0.8em 0; display: block; }}
blockquote {{ border-left: 3px solid #1abc9c; padding-left: 1em; color: #666; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 0.4em; font-size: 0.9em; }}
th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>{article_title}</h1>
{body_content}
</body>
</html>"""

    return article_title, styled_html, images


def clean_title(title):
    title = re.sub(r"_For_group_share.*$", "", title)
    title = re.sub(r"【更多IT资源.*?】", "", title)
    return title.strip()


def create_epub():
    global global_img_counter
    global_img_counter = 0

    print("=== 步骤1: 收集和下载图片 ===")
    all_html_data = []
    online_urls = []

    for chapter_dir_name in CHAPTER_DIRS:
        chapter_path = DOCS_DIR / chapter_dir_name
        if not chapter_path.exists():
            continue
        for html_file in sorted(chapter_path.glob("*.html"), key=lambda f: get_chapter_order(f.name)):
            all_html_data.append((chapter_dir_name, html_file))
            with open(html_file, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml")
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if src.startswith("http://") or src.startswith("https://"):
                    online_urls.append(src)

    downloaded = batch_download_images(online_urls)

    print("\n=== 步骤2: 构建EPUB ===")
    book = epub.EpubBook()
    book.set_identifier("sparrow-recsys-deep-learning-2024")
    book.set_title(BOOK_TITLE)
    book.set_language("zh")
    book.add_author(BOOK_AUTHOR)

    spine = ["nav"]
    toc = []
    all_image_items = []
    chapter_num = 0

    for chapter_dir_name in CHAPTER_DIRS:
        chapter_path = DOCS_DIR / chapter_dir_name
        if not chapter_path.exists():
            continue

        chapter_num += 1
        chapter_name = re.sub(r"^\d+、", "", chapter_dir_name)
        chapter_items = []

        html_files = sorted(
            chapter_path.glob("*.html"),
            key=lambda f: get_chapter_order(f.name),
        )

        for html_file in html_files:
            title, content_html, images = extract_content(str(html_file), downloaded)
            clean_t = clean_title(title)

            ch = epub.EpubHtml(
                title=clean_t,
                file_name=f"chapter_{chapter_num:02d}_{len(chapter_items):02d}.xhtml",
                lang="zh",
            )
            ch.set_content(content_html.encode("utf-8"))
            book.add_item(ch)
            spine.append(ch)
            chapter_items.append(ch)

            for img_name, img_data, img_mime in images:
                img_item = epub.EpubItem(uid=img_name, file_name=f"images/{img_name}", media_type=img_mime, content=img_data)
                book.add_item(img_item)
                all_image_items.append(img_item)

            print(f"  {clean_t} ({len(images)}张图)")

        if chapter_items:
            toc.append((epub.Section(chapter_name), chapter_items))

    book.toc = toc
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(OUTPUT_FILE), book)
    print(f"\n=== 完成 ===")
    print(f"EPUB: {OUTPUT_FILE}")
    print(f"共 {chapter_num} 章, {len(spine)-1} 篇, {len(all_image_items)} 张图片")


if __name__ == "__main__":
    create_epub()
