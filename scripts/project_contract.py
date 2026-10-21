"""Portable developer checks. Canonical template: software-development skill.

Generated project copies are hash-locked development tools, never runtime imports.
Checks inspect source without importing scientific modules. External checks run
only for verify (discover validates source/configuration but starts no tests).
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

ENGINE_VERSION = 1


class ContractError(ValueError):
    pass


def read_document(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise ContractError("YAML needs PyYAML; install scripts/contracts-requirements.txt") from error
        return yaml.safe_load(text)


def local(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or re.match(r"^[A-Za-z]:", name):
        raise ContractError(f"path must be project-relative: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ContractError(f"path escapes project: {name}")
    return path


def text_file(root, name):
    path = local(root, name)
    if not path.is_file():
        raise ContractError(f"missing file: {name}")
    return path.read_text(encoding="utf-8-sig")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(root, schema=None):
    import jsonschema
    contract = read_document(root / "architecture.yaml")
    if schema is None:
        schema = root / "architecture.schema.json"
    definition = read_document(schema)
    jsonschema.Draft202012Validator.check_schema(definition)
    errors = sorted(jsonschema.Draft202012Validator(definition).iter_errors(contract), key=lambda e: str(e.path))
    if errors:
        raise ContractError("; ".join(f"{'/'.join(map(str, e.path)) or 'contract'}: {e.message}" for e in errors))
    for field in ("boundaries", "components", "checks"):
        ids = [entry["id"] for entry in contract.get(field, [])]
        if len(ids) != len(set(ids)):
            raise ContractError(f"duplicate {field} identifiers")
    if not any("fast" in row["groups"] for row in contract["checks"]):
        raise ContractError("at least one required fast check must be declared")
    return contract


def imported_names(tree, module, is_package=False):
    package = module.split(".") if is_package else module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package):
                    raise ContractError(f"relative import escapes package: {module}:{node.lineno}")
                base = ".".join(package[:len(package) - node.level + 1] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                yield node.lineno, base
            for alias in node.names:
                if alias.name != "*":
                    yield node.lineno, f"{base}.{alias.name}".strip(".")
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            function = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if function in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
                name = node.args[0].value
                if not name.startswith("."):
                    yield node.lineno, name


def check_boundaries(root, contract):
    issues = []
    source_root = local(root, contract.get("source_root", "src"))
    for rule in contract.get("boundaries", []):
        files = set()
        for pattern in rule["sources"]:
            local(root, pattern)
            matches = list(root.glob(pattern))
            if not matches:
                issues.append(f"{rule['id']}: source pattern matched nothing: {pattern}")
            files.update(path.resolve() for path in matches if path.is_file())
        used_exceptions = set()
        for path in sorted(files):
            if path.suffix != ".py" or not path.is_relative_to(source_root):
                issues.append(f"{rule['id']}: expected Python source under {source_root.name}: {path.relative_to(root)}")
                continue
            rel = path.relative_to(root).as_posix()
            mod = path.relative_to(source_root).with_suffix("").as_posix().replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[:-9]
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
            for line, name in imported_names(tree, mod, path.name == "__init__.py"):
                if not any(name == prefix or name.startswith(prefix + ".") for prefix in rule["forbidden"]):
                    continue
                exceptions = [i for i, item in enumerate(rule.get("exceptions", []))
                              if item["path"] == rel and (name == item["import"] or name.startswith(item["import"] + "."))]
                if exceptions:
                    used_exceptions.update(exceptions)
                else:
                    issues.append(f"{rule['id']}: {rel}:{line} imports forbidden {name}")
        for index, exception in enumerate(rule.get("exceptions", [])):
            if index not in used_exceptions:
                issues.append(f"{rule['id']}: unused exception {exception['path']} -> {exception['import']}")
    return issues


def check_components(root, contract):
    issues = []
    for item in contract.get("components", []):
        source = text_file(root, item["path"])
        if item["kind"] == "python":
            names = {node.name for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            exists = item["symbol"] in names
        else:
            source = re.sub(r"/\*[\s\S]*?\*/|^\s*//[^\n]*", "", source, flags=re.MULTILINE)
            symbol = re.escape(item["symbol"])
            exists = bool(re.search(rf"\b(?:function\s+{symbol}\s*\(|(?:const|let|var)\s+{symbol}\s*=)", source))
        if not exists:
            issues.append(f"{item['id']}: missing defined symbol {item['symbol']} in {item['path']}")
        for example in item["examples"]:
            text_file(root, example)
        for use in item["uses"]:
            code = text_file(root, use["path"])
            code = re.sub(r"/\*[\s\S]*?\*/|^\s*//[^\n]*", "", code, flags=re.MULTILINE)
            if not re.search(rf"(?<![\w$]){re.escape(use['call'])}\s*\(", code):
                issues.append(f"{item['id']}: missing required call {use['call']} in {use['path']}")
    return issues


def check_theme(root, contract):
    theme = contract.get("theme")
    if not theme:
        return []
    import tinycss2
    tokens = set()
    for name in theme["files"]:
        tokens.update(re.findall(r"(--[\w-]+)\s*:", text_file(root, name)))
    tokens.update(row["name"] for row in theme.get("runtime_tokens", []))
    issues = [f"theme: missing required token {name}" for name in theme["required_tokens"] if name not in tokens]
    def blocks(items):
        for block in items:
            yield block
            if block.type == "at-rule" and block.content is not None and block.lower_at_keyword in {"media", "supports", "layer", "container", "scope"}:
                yield from blocks(tinycss2.parse_rule_list(block.content, skip_comments=True, skip_whitespace=True))

    def values(items):
        for token in items:
            yield token
            children = getattr(token, "arguments", getattr(token, "content", None))
            if children is not None:
                yield from values(children)

    for rule in theme.get("rules", []):
        stylesheet = tinycss2.parse_stylesheet(text_file(root, rule["path"]), skip_comments=True, skip_whitespace=True)
        selected = set()
        for block in blocks(stylesheet):
            if block.type == "error":
                issues.append(f"theme: {rule['path']}:{block.source_line}: {block.message}")
                continue
            if block.type != "qualified-rule":
                continue
            selector = tinycss2.serialize(block.prelude).strip()
            wanted = [s for s in rule["selectors"] if s in [piece.strip() for piece in selector.split(",")]]
            if not wanted:
                continue
            selected.update(wanted)
            for decl in tinycss2.parse_declaration_list(block.content, skip_comments=True, skip_whitespace=True):
                if decl.type == "error":
                    issues.append(f"theme: invalid declaration in {selector}")
                    continue
                if decl.type != "declaration":
                    continue
                value = tinycss2.serialize(decl.value)
                for token in re.findall(r"var\(\s*(--[\w-]+)", value):
                    if token not in tokens:
                        issues.append(f"theme: {selector} references undefined {token}")
                literal_colour = any(t.type == "hash" or (t.type == "function" and t.lower_name in {"rgb", "rgba", "hsl", "hsla", "color", "oklch", "oklab", "lab", "lch"}) for t in values(decl.value))
                if "color" in decl.lower_name or decl.lower_name in {"background", "border", "outline", "box-shadow", "fill", "stroke"}:
                    from tinycss2.color3 import parse_color
                    literal_colour = literal_colour or any(t.type == "ident" and t.lower_value not in {"transparent", "currentcolor"} and parse_color(t) is not None for t in values(decl.value))
                if rule.get("forbid_raw_colours") and literal_colour:
                    issues.append(f"theme: {selector} {decl.name} uses a raw colour")
                layout = re.match(r"^(?:width|height|min-width|max-width|min-height|max-height|padding|margin|gap|row-gap|column-gap|font-size)(?:-|$)", decl.name)
                if rule.get("forbid_fixed_layout") and layout and any(t.type == "dimension" and t.lower_unit == "px" for t in values(decl.value)):
                    issues.append(f"theme: {selector} {decl.name} uses fixed layout pixels")
        for selector in set(rule["selectors"]) - selected:
            issues.append(f"theme: missing checked selector {selector} in {rule['path']}")
    return issues


def check_content(root, contract):
    spec = contract.get("content")
    if not spec:
        return []
    rows = read_document(local(root, spec["path"]))
    if not isinstance(rows, list) or not rows:
        return ["content: expected a nonempty list"]
    issues, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(f"content row {index}: expected object")
            continue
        for field in spec["required_fields"]:
            if not isinstance(row.get(field), str) or not row[field].strip():
                issues.append(f"content row {index}: missing {field}")
        key = row.get(spec["unique_key"])
        if not isinstance(key, str) or key in seen:
            issues.append(f"content row {index}: invalid or duplicate {spec['unique_key']}")
        else:
            seen.add(key)
        for field in spec.get("link_fields", []):
            value = row.get(field, "")
            if not isinstance(value, str) or not value:
                issues.append(f"content row {index}: invalid link {field}")
                continue
            parts = urlsplit(value)
            if parts.scheme:
                if parts.scheme not in {"https", "http"} or not parts.netloc:
                    issues.append(f"content row {index}: unsupported link {field}")
            elif not parts.path or not local(root, parts.path.lstrip("/")).is_file():
                issues.append(f"content row {index}: missing internal destination {field}")
    return issues


def static_issues(root, contract):
    issues = []
    for name in contract["required_paths"]:
        path = local(root, name)
        if not path.exists() or (path.is_dir() and not any(path.iterdir())):
            issues.append(f"missing or empty required path: {name}")
    lock = root / "architecture.lock.json"
    if lock.is_file():
        data = read_document(lock)
        if data.get("engine_version") != ENGINE_VERSION:
            issues.append("generated engine version differs from lock")
        for name, expected in data.get("files", {}).items():
            path = local(root, name)
            if not path.is_file() or digest(path) != expected:
                issues.append(f"generated tooling changed: {name}; update from canonical template with a reason")
    for checker in (check_boundaries, check_components, check_theme, check_content):
        try:
            issues.extend(checker(root, contract))
        except (OSError, ValueError, SyntaxError) as error:
            issues.append(f"{checker.__name__}: {error}")
    return issues


def run_check(root, check):
    started = time.monotonic()
    result = {"id": check["id"], "passed": False}
    argv = [sys.executable if value == "{python}" else value for value in check["argv"]]
    with tempfile.TemporaryDirectory(prefix="project-contract-") as temporary:
        report = Path(temporary) / "tests.xml"
        if check["kind"] == "pytest":
            if not (len(argv) >= 3 and argv[1:3] == ["-m", "pytest"]):
                raise ContractError(f"{check['id']}: pytest check must use {{python}} -m pytest")
            if any("junit" in arg for arg in argv):
                raise ContractError(f"{check['id']}: verifier owns the temporary JUnit report")
            argv.append(f"--junitxml={report}")
        try:
            process = subprocess.Popen(argv, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            try:
                stdout, stderr = process.communicate(timeout=check["timeout_seconds"])
            except subprocess.TimeoutExpired:
                # Only the process tree this check created is stopped. Children
                # must not keep the verifier's pipes open after its deadline.
                if os.name == "nt":
                    try:
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)
                raise
            result["returncode"] = process.returncode
            output = stdout + stderr
            result["output"] = output[-6000:]
            result["passed"] = process.returncode == 0
            if check["kind"] == "pytest":
                if not report.is_file():
                    result.update(passed=False, evidence="pytest produced no result report")
                else:
                    tree = ET.parse(report)
                    cases = tree.findall(".//testcase")
                    skipped = sum(case.find("skipped") is not None for case in cases)
                    failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
                    passed = len(cases) - skipped - failed
                    result.update(tests=len(cases), skipped=skipped, executed_passes=passed)
                    if passed < check["min_passed"] or (skipped and not check.get("allow_skips", False)) or failed:
                        result.update(passed=False, evidence="insufficient executed tests, failed tests or required tests skipped")
        except subprocess.TimeoutExpired:
            result.update(evidence=f"timeout after {check['timeout_seconds']} seconds")
        except (OSError, ET.ParseError) as error:
            result.update(evidence=str(error))
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def main(argv=None, default_root=None, default_schema=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["verify", "discover"], default="verify")
    parser.add_argument("--root", type=Path, default=default_root or Path.cwd())
    parser.add_argument("--schema", type=Path, default=default_schema)
    parser.add_argument("--structure", action="store_true", help="validate configuration/source only; do not run declared checks")
    parser.add_argument("--group", choices=["fast", "release"], default="fast")
    parser.add_argument("--check", action="append", dest="selected")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        contract = load_contract(root, args.schema)
        issues = static_issues(root, contract)
        report = {"project": contract["project"], "engine_version": ENGINE_VERSION, "date": datetime.now(timezone.utc).isoformat(), "issues": issues, "checks": [], "structure_only": args.structure, "ok": False}
        if args.command == "discover":
            report["components"] = contract.get("components", []) if not issues else []
        elif not args.structure and not issues:
            checks = [row for row in contract["checks"] if args.group in row["groups"]]
            if args.selected:
                known = {row["id"] for row in checks}
                missing = set(args.selected) - known
                if missing:
                    raise ContractError(f"unknown checks in {args.group}: {sorted(missing)}")
                checks = [row for row in checks if row["id"] in args.selected]
            if not checks:
                raise ContractError(f"no required checks selected for {args.group}")
            for check in checks:
                result = run_check(root, check)
                report["checks"].append(result)
                if not args.json:
                    print(f"{'PASS' if result['passed'] else 'FAIL'} {result['id']} ({result['seconds']}s)", flush=True)
        report["ok"] = not issues and all(row["passed"] for row in report["checks"])
        if args.json or args.command == "discover":
            print(json.dumps(report, indent=2))
        else:
            for issue in issues:
                print(f"FAIL {issue}")
            for row in report["checks"]:
                if not row["passed"]:
                    print(row.get("evidence", ""))
                    print(row.get("output", ""))
            print(f"{'PASS' if report['ok'] else 'FAIL'} {contract['project']} ({'source/configuration only' if args.structure else args.group})")
        return 0 if report["ok"] else 1
    except (OSError, ValueError, ImportError, KeyError) as error:
        if args.json:
            print(json.dumps({"ok": False, "issues": [str(error)]}))
        else:
            print(f"FAIL {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
