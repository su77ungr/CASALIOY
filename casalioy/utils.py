"""some useful functions"""
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.styles import Style
from pyexpat import ExpatError
from requests import HTTPError

style = Style.from_dict(
    {
        "r": "italic gray",  # remark
        "w": "italic yellow",  # warning
        "d": "bold red",  # danger
        "b": "bold",
        "i": "italic",
        "question": "ansicyan",
        "answer": "ansigreen",
        "source": "ansimagenta",
    }
)


def escape_for_html(text, **kwargs) -> str:
    """escape unicode stuff. kwargs are changed in-place."""
    escape_one = lambda v: v.replace("\f", " ").replace("\b", "\\")
    for k, v in kwargs.items():
        kwargs[k] = escape_one(str(v))
    text = escape_one(text)
    return text


def print_HTML(text: str, **kwargs) -> None:
    """print formatted HTML text"""
    try:
        text = escape_for_html(text, **kwargs)
        print_formatted_text(HTML(text).format(**kwargs), style=style)
    except ExpatError:
        print(text.format(**kwargs))


def prompt_HTML(session: PromptSession, prompt: str, **kwargs) -> str:
    """print formatted HTML text"""
    try:
        prompt = escape_for_html(prompt, **kwargs)
        return session.prompt(HTML(prompt).format(**kwargs), style=style)
    except ExpatError:
        return input(prompt.format(**kwargs))


def download_if_repo(path: str) -> str:
    """download model from HF if not local"""
    # check if dataset
    split = path.split("/")
    is_dataset = split[0] == "datasets"
    is_file = path.endswith(".bin")
    allow_patterns = split[-1] if is_file else ["*.bin", "*.json"]
    repo_id = path
    if is_dataset:
        split = split[1:]
        repo_id = "/".join(split)
    if path.endswith(".bin"):
        repo_id = "/".join(split[:2])

    p = Path(path) if path.startswith("models/") else "models" / Path(path)
    if (is_file and p.is_file()) or (not is_file and p.is_dir()):
        print_HTML(f"<r>found local model {'file' if is_file else 'dir'} at {p}</r>")
        return str(p)

    try:
        validate_repo_id(repo_id)
        print_HTML("<r>Downloading {model_type} {model} from HF</r>", model=path, model_type="dataset" if is_dataset else "model")
        new_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                local_dir=str(p.parent if is_file else p),
                repo_type="dataset" if is_dataset else None,
                local_dir_use_symlinks=False,
            )
        )
        return str(new_path.resolve())

    except (HFValidationError, HTTPError) as e:
        print_HTML("<w>Could not download model {model} from HF: {e}</w>", model=path, e=e)
