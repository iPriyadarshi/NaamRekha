"""Stage the project website source for MkDocs (see mkdocs.yml).

The docs link freely to results/, notebooks/ and scripts/ with repo-relative paths.
MkDocs only serves files inside its docs_dir, so this copies the parts of the repo the
site shows into build/site_src/, keeping their repo layout so those relative links
still resolve:

  README.md          -> index.md (the home page)
  docs/              -> docs/
  results/           -> results/ (CSVs, figures and outputs the docs link to)
  report/REPORT.pdf  -> report/REPORT.pdf

Links to anything else in the repo (notebooks, scripts, src, folders) are rewritten
to point at the file on GitHub.

Run from the repo root:  uv run --only-group docs python scripts/build_site.py
then:                    uv run --only-group docs mkdocs build --strict
"""

from __future__ import annotations

import os
import posixpath
import re
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "build", "site_src")
GITHUB = "https://github.com/iPriyadarshi/NaamRekha"
BRANCH = "main"

COPY_TREES = ["docs", "results"]
COPY_FILES = {"README.md": "index.md", "report/REPORT.pdf": "report/REPORT.pdf"}

REPORT_BANNER = (
    "> **Read the report:** [web version](docs/phase6/report.md) · "
    "[PDF](report/REPORT.pdf)\n\n"
)

# [text](target) where target has no spaces; images match too via the leading "!".
LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def rewrite(target: str, page: str) -> str:
    """Map one link target on a page (repo-relative path) to its site target."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
        return target
    path, _, anchor = target.partition("#")
    if not path:
        return target
    repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    if repo_path.startswith(".."):
        return target
    staged = os.path.join(OUT, *repo_path.split("/"))
    if os.path.isfile(staged):
        return target
    kind = "tree" if os.path.isdir(os.path.join(REPO_ROOT, repo_path)) else "blob"
    url = f"{GITHUB}/{kind}/{BRANCH}/{repo_path}"
    return f"{url}#{anchor}" if anchor else url


def main() -> None:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for tree in COPY_TREES:
        shutil.copytree(os.path.join(REPO_ROOT, tree), os.path.join(OUT, tree))
    for src, dst in COPY_FILES.items():
        os.makedirs(os.path.dirname(os.path.join(OUT, dst)) or OUT, exist_ok=True)
        shutil.copyfile(os.path.join(REPO_ROOT, src), os.path.join(OUT, dst))

    # Repo-relative path of each staged page, for resolving its links.
    pages = {"index.md": "README.md"}
    for dirpath, _, filenames in os.walk(os.path.join(OUT, "docs")):
        for name in filenames:
            if name.endswith(".md"):
                rel = os.path.relpath(os.path.join(dirpath, name), OUT).replace(os.sep, "/")
                pages[rel] = rel

    for staged, page in pages.items():
        full = os.path.join(OUT, staged)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        text = LINK.sub(lambda m: m[1] + rewrite(m[2], page) + m[3], text)
        if staged == "index.md":
            # Home page: README links like docs/README.md resolve from the root as-is.
            title, _, rest = text.partition("\n")
            text = f"{title}\n\n{REPORT_BANNER}{rest.lstrip()}"
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    print(f"staged site source in {os.path.relpath(OUT, REPO_ROOT)}")


if __name__ == "__main__":
    main()
