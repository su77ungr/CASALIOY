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


def download_if_repo(path: str, file: str = None, allow_patterns: str | list[str] = None) -> str:
    """download model from HF if not local"""
    if allow_patterns is None:
        allow_patterns = ["*.bin", "*.json"]

    # check if dataset
    split = path.split("/")
    is_dataset = split[0] == "datasets"
    if is_dataset:
        split = split[1:]
        path = "/".join(split)

    p = "models/datasets" / Path(path) if is_dataset else "models" / Path(path)
    if p.is_file() or p.is_dir():
        print_HTML(f"<r>found local model at {p}</r>")
        return str(p)

    try:
        if path.endswith(".bin"):
            path, file = "/".join(split[: 3 if is_dataset else 2]), split[-1]
        validate_repo_id(path)
        print_HTML("<r>Downloading {model_type} {model} from HF</r>", model=path, model_type="dataset" if is_dataset else "model")
        new_path = Path(
            snapshot_download(
                repo_id=path,
                allow_patterns=file or allow_patterns,
                local_dir=str(p),
                repo_type="dataset" if is_dataset else None,
                local_dir_use_symlinks=False,
            )
        )
        if file is not None:
            files = [f for f in new_path.iterdir() if f.is_file() and f.name.endswith(".bin")]
            if len(files) > 1:
                names = "\n".join([f" - {f.name}" for f in files])
                raise ValueError(f"Multiple model files found: \n\n{names}\n\n")
            new_path = files[0]
        return str(new_path.resolve())

    except (HFValidationError, HTTPError) as e:
        print_HTML("<w>Could not download model {model} from HF: {e}</w>", model=path, e=e)
