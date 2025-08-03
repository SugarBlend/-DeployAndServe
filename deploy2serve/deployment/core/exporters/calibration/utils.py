from typing import Dict, Callable, Union, List, Optional
from pathlib import Path, PurePosixPath


class Uncompress:
    _formats: Dict[str, Callable] = {}

    @classmethod
    def register(cls, extensions: Union[str, List[str]]):
        def decorator(executor_cls: Callable):
            if isinstance(extensions, str):
                cls._formats[extensions] = executor_cls
            else:
                for ext in extensions:
                    cls._formats[ext] = executor_cls
            return executor_cls
        return decorator

    @classmethod
    def uncompress(cls, archive_path: Union[Path, str], output_dir: Union[Path, str]) -> str:
        extension = Path(archive_path).suffix
        uncompress_func = cls._formats.get(extension)
        if not uncompress_func:
            raise ValueError(f"Unsupported extension type: {extension}")
        return uncompress_func(archive_path, output_dir)


@Uncompress.register(".zip")
def uncompress_zip(archive_path: str, output_dir: str) -> str:
    import zipfile
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
        names = zip_ref.namelist()
        paths = [PurePosixPath(name) for name in names if not name.endswith('/')]
        if not paths:
            return str()
        root_dirs = {p.parts[0] for p in paths}
        if len(root_dirs) == 1:
            return root_dirs.pop()
        return str()


# @Uncompress.register(".7z")
# def uncompress_7z(archive_path: str, output_dir: str) -> None:
#     import py7zr
#     with py7zr.SevenZipFile(archive_path, mode='r') as z:
#         z.extractall(path=output_dir)
#         print(f'7z-архив {archive_path} успешно распакован в {output_dir}')
#
#
# @Uncompress.register([".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz"])
# def uncompress_tar(archive_path: str, output_dir: str) -> None:
#     import tarfile
#     with tarfile.open(archive_path, "r:*") as tar:
#         tar.extractall(path=output_dir)
