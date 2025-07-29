import mimetypes


def is_image_file(filepath: str) -> bool:
    mime = mimetypes.guess_type(filepath)[0]
    return mime is not None and mime.startswith("image/")


def is_video_file(filepath: str) -> bool:
    mime = mimetypes.guess_type(filepath)[0]
    return mime is not None and mime.startswith("video/")
