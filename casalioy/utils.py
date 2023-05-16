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


def print_HTML(text: str, **kwargs) -> None:
    """print formatted HTML text"""
    try:
        print_formatted_text(HTML(text).format(**kwargs), style=style)
    except ExpatError:
        print(text)


def prompt_HTML(session: PromptSession, prompt: str, **kwargs) -> str:
    """print formatted HTML text"""
    try:
        return session.prompt(HTML(prompt).format(**kwargs), style=style)
    except ExpatError:
        return input(prompt)


def download_if_repo(path: str, file: str = None, allow_patterns: str | list[str] = "*.bin") -> str:
    """download model from HF if not local"""
    p = Path(path)
    if p.is_file() or p.is_dir():
        return str(p)
    try:
        if path.endswith(".bin"):
            split = path.split("/")
            path, file = "/".join(split[:2]), split[-1]
        validate_repo_id(path)
        print_HTML("<r>Downloading {model} from HF</r>", model=path)
        new_path = Path(snapshot_download(repo_id=path, allow_patterns=file or allow_patterns))
        if file is not None:
            files = [f for f in new_path.iterdir() if f.is_file() and f.name.endswith(".bin")]
            if len(files) > 1:
                names = "\n".join([f" - {f.name}" for f in files])
                raise ValueError(f"Multiple model files found: \n\n{names}\n\n")
            new_path = files[0]
        return str(new_path.resolve())

    except (HFValidationError, HTTPError):
        print_HTML("<w>Could not download model {model} from HF</w>", model=path)
