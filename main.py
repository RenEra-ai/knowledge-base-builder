import json
import time
import re
import os
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

with open('config.json', 'r') as f:
    data = json.load(f)

urls = data['urls']
output_folder = 'knowledge_base'

client = httpx.Client(
    follow_redirects=True,
    timeout=30.0,
    headers={
        "User-Agent": "KnowledgeBaseBuilder/2.0 (Boomi Documentation Indexer)"
    }
)

failed_urls = []
DELAY_BETWEEN_REQUESTS = 1.0


def count_urls(url_list):
    total = 0
    for obj in url_list:
        total += 1
        total += count_urls(obj.get('children', []))
    return total


total_urls = count_urls(urls)
processed = 0


def fetch_html(url, retries=3):
    time.sleep(DELAY_BETWEEN_REQUESTS)
    for attempt in range(retries):
        try:
            response = client.get(url)
            if response.status_code == 404:
                print(f"  WARNING: 404 for {url}")
                failed_urls.append(url)
                return ""
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} for {url} (waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {url} — {e}")
                failed_urls.append(url)
                return ""


def extract_article_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    content_div = soup.find('div', class_='theme-doc-markdown markdown')
    return str(content_div) if content_div else ""


def fix_relative_urls(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a['href']
        parsed_href = urlparse(href)
        if not parsed_href.scheme and not parsed_href.netloc:
            a['href'] = urljoin(base_url, href)
        else:
            a['href'] = href
    for img in soup.find_all('img', src=True):
        src = img['src']
        if src.startswith('/'):
            img['src'] = urljoin(base_url, src)
    return str(soup)


def remove_class_id_and_svg(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup.find_all(True):
        if 'class' in tag.attrs:
            del tag.attrs['class']
        if 'id' in tag.attrs:
            del tag.attrs['id']
    for svg in soup.find_all('svg'):
        svg.decompose()
    return str(soup)


def build_breadcrumbs(path):
    return " > ".join(f"<a href='{url}'>{title}</a>" for title, url in path)


def sanitize_filename(title):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', title) + '.html'


def process_url(url_obj, path, indent_level=0):
    global processed
    processed += 1
    url = url_obj['url']
    print(f"[{processed}/{total_urls}] Fetching: {url[:80]}...")

    parsed = urlparse(url)
    page_base_url = f"{parsed.scheme}://{parsed.netloc}"

    html = fetch_html(url)
    article_content = extract_article_content(html)
    article_content = fix_relative_urls(article_content, page_base_url)
    article_content = remove_class_id_and_svg(article_content)

    soup = BeautifulSoup(article_content, 'html.parser')
    first_heading = soup.find('h1')
    title = first_heading.text if first_heading else url
    filename = sanitize_filename(title)

    path.append((title, url))
    breadcrumbs = build_breadcrumbs(path)
    html_output = f"<h{indent_level + 1}>{title}</h{indent_level + 1}>\n\n"
    html_output += f"<p><strong>Path:</strong> {breadcrumbs}</p>\n\n"

    if first_heading:
        first_heading.extract()
    content_html = str(soup)
    html_output += content_html + "\n\n"

    for child in url_obj.get('children', []):
        _, child_content = process_url(child, path[:], indent_level + 1)
        html_output += child_content

    path.pop()
    return filename, html_output


os.makedirs(output_folder, exist_ok=True)

try:
    for url_obj in urls:
        filename, content = process_url(url_obj, [])
        file_path = os.path.join(output_folder, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
finally:
    client.close()
    if failed_urls:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(failed_urls)} URLs failed:")
        for u in failed_urls:
            print(f"  - {u}")
        with open(os.path.join(output_folder, '_failed_urls.txt'), 'w') as f:
            f.write('\n'.join(failed_urls))

print("HTML files generated successfully")
