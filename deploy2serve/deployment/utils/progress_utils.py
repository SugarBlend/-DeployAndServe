from colorama import Fore
from typing import Dict, Any


def get_progress_options() -> Dict[str, Any]:
    custom_format = (
        f"{Fore.WHITE}{{desc}}: {{percentage:2.0f}}% {Fore.LIGHTGREEN_EX}|{{bar}}| "
        f"{Fore.WHITE}{{n_fmt}}/{{total_fmt}} [{Fore.LIGHTBLUE_EX}{{elapsed}}<{{remaining}} "
        f"{{rate_fmt}}]{Fore.RESET}"
    )
    progress_options = {
        "bar_format": custom_format, "position": 0, "leave": True, "ncols": 75, "colour": None,
    }
    return progress_options
