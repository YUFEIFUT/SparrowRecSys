#!/usr/bin/env python3
"""将3个特殊HTML文件生成独立EPUB，只保留正文+图片+评论。"""

import os
import re
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from ebooklib import epub
import urllib.request

DOCS_DIR = Path(__file__).parent / "docs"
OUTPUT_FILE = Path(__file__).parent / "深度学习推荐系统实践_剩余部分.epub"

PROBLEM_FILES = [
    r"docs\3、特征工程\07 _ Embedding进阶：如何利用图结构数据生成Graph Embedding？.html",
    r"docs\5、推荐模型\15 _ 协同过滤：最经典的推荐模型，我们应该掌握什么？.html",
    r"docs\6、模型评估\25 _ 评估指标：我们可以用哪些指标来衡量模型的好坏？.html",
]

BOOK_TITLE = "深度学习推荐系统实践（剩余部分）"
BOOK_AUTHOR = "王喆"

global_img_counter = 0


def download_one_image(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "png" in ct: mime, ext = "image/png", "png"
            elif "gif" in ct: mime, ext = "image/gif", "gif"
            elif "webp" in ct: mime, ext = "image/webp", "webp"
            else: mime, ext = "image/jpeg", "jpg"
            return url, data, mime, ext
    except Exception:
        return url, None, None, None


def batch_download_images(urls):
    results = {}
    unique = list(set(urls))
    print(f"  下载 {len(unique)} 张在线图片...")
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(download_one_image, u): u for u in unique}
        done = 0
        for fut in as_completed(futures):
            url, data, mime, ext = fut.result()
            done += 1
            if data: results[url] = (data, mime, ext)
            if done % 50 == 0 or done == len(unique):
                print(f"    {done}/{len(unique)}")
    print(f"  完成: {len(results)}/{len(unique)}")
    return results


def process_images(element, html_dir, downloaded):
    """处理元素内所有图片，返回 (images_list, modified_element)。"""
    global global_img_counter
    images = []

    for img in element.find_all("img"):
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
                global_img_counter += 1
                img_name = f"img_{global_img_counter}.{ext}"
                img["src"] = f"images/{img_name}"
                images.append((img_name, data, mime))
            continue

        # 在线URL
        if src.startswith("http://") or src.startswith("https://"):
            if src in downloaded:
                data, mime, ext = downloaded[src]
                global_img_counter += 1
                img_name = f"img_{global_img_counter}.{ext}"
                img["src"] = f"images/{img_name}"
                images.append((img_name, data, mime))
            else:
                img.decompose()
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
            global_img_counter += 1
            img_name = f"img_{global_img_counter}.{ext}"
            img["src"] = f"images/{img_name}"
            images.append((img_name, data, mime))

    return images


def extract_content(html_path, downloaded):
    global global_img_counter

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    body = soup.find("body")
    html_dir = Path(html_path).parent

    # ===== 1. 提取标题 =====
    h1 = body.find("h1")
    article_title = h1.get_text(strip=True) if h1 else Path(html_path).stem

    # ===== 2. 提取正文（slate编辑器内容）=====
    slate_div = body.find("div", attrs={"data-slate-editor": True})
    if slate_div:
        # 清理slate内容：移除script/style等
        for tag in slate_div.find_all(["script", "style", "noscript"]):
            tag.decompose()
        # 处理图片
        article_images = process_images(slate_div, html_dir, downloaded)
        article_html = str(slate_div)
    else:
        article_images = []
        article_html = "<p>未找到正文内容</p>"

    # ===== 3. 提取评论 =====
    comments_html = ""
    comments_images = []
    comments_div = body.find("div", class_=re.compile(r"_3-W_zrq4_0"))
    if comments_div:
        # 清理评论区
        for tag in comments_div.find_all(["script", "style", "noscript", "textarea"]):
            tag.decompose()
        # 移除"提交留言"按钮等
        for div in comments_div.find_all("div"):
            text = div.get_text(strip=True)
            if text in ("提交留言", "Ctrl + Enter 发表", "Ctrl + Enter 发表"):
                div.decompose()
        # 处理图片
        comments_images = process_images(comments_div, html_dir, downloaded)
        comments_html = str(comments_div)

    all_images = article_images + comments_images

    # ===== 4. 构建EPUB页面 =====
    styled_html = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>{article_title}</title>
<meta charset="utf-8"/>
<style>
body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.8; color: #333; padding: 0.5em; }}
h1 {{ font-size: 1.5em; margin-bottom: 0.5em; color: #1a1a1a; }}
h2 {{ font-size: 1.2em; margin-top: 1.2em; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
h3 {{ font-size: 1.1em; color: #34495e; }}
p {{ margin: 0.6em 0; text-align: justify; }}
strong {{ color: #c0392b; }}
img {{ max-width: 100%; height: auto; margin: 0.8em 0; display: block; }}
blockquote {{ border-left: 3px solid #1abc9c; padding-left: 1em; color: #666; margin: 1em 0; }}
code {{ background: #f8f8f8; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
pre {{ background: #f8f8f8; padding: 1em; overflow-x: auto; border-radius: 3px; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 0.4em; font-size: 0.9em; }}
th {{ background: #f5f5f5; }}
.comments-section {{ margin-top: 2em; padding-top: 1em; border-top: 2px solid #eee; }}
.comments-section h2 {{ color: #666; }}
.comment-item {{ margin-bottom: 1em; padding-bottom: 1em; border-bottom: 1px solid #f0f0f0; }}
.comment-author {{ font-weight: bold; color: #333; }}
.comment-date {{ color: #999; font-size: 0.85em; }}
.comment-text {{ margin-top: 0.3em; }}
.comment-reply {{ margin-top: 0.5em; padding-left: 1em; color: #666; border-left: 2px solid #ddd; }}
</style>
</head>
<body>
<h1>{article_title}</h1>
{article_html}
{f'<div class="comments-section">{comments_html}</div>' if comments_html else ''}
</body>
</html>"""

    return article_title, styled_html, all_images


def clean_title(title):
    title = re.sub(r"_For_group_share.*$", "", title)
    title = re.sub(r"【更多IT资源.*?】", "", title)
    return title.strip()


def create_epub():
    global global_img_counter
    global_img_counter = 0

    # 收集在线图片URL
    online_urls = []
    for fp in PROBLEM_FILES:
        full_path = Path(__file__).parent / fp
        with open(full_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src.startswith("http://") or src.startswith("https://"):
                online_urls.append(src)

    downloaded = batch_download_images(online_urls)

    # 构建EPUB
    book = epub.EpubBook()
    book.set_identifier("sparrow-recsys-extra-2024")
    book.set_title(BOOK_TITLE)
    book.set_language("zh")
    book.add_author(BOOK_AUTHOR)

    spine = ["nav"]
    toc = []
    all_image_items = []

    for i, fp in enumerate(PROBLEM_FILES):
        full_path = Path(__file__).parent / fp
        title, content_html, images = extract_content(str(full_path), downloaded)
        clean_t = clean_title(title)

        ch = epub.EpubHtml(
            title=clean_t,
            file_name=f"chapter_{i:02d}.xhtml",
            lang="zh",
        )
        ch.set_content(content_html.encode("utf-8"))
        book.add_item(ch)
        spine.append(ch)
        toc.append(ch)

        for img_name, img_data, img_mime in images:
            img_item = epub.EpubItem(uid=img_name, file_name=f"images/{img_name}", media_type=img_mime, content=img_data)
            book.add_item(img_item)
            all_image_items.append(img_item)

        print(f"  {clean_t} ({len(images)}张图)")

    book.toc = [(epub.Section("剩余部分"), toc)]
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(OUTPUT_FILE), book)
    print(f"\nEPUB: {OUTPUT_FILE}")
    print(f"共 3 篇, {len(all_image_items)} 张图片")


if __name__ == "__main__":
    create_epub()
