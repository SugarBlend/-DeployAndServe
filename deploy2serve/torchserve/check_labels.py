from triton.custom.yolo.yolo import check_labels


if __name__ == "__main__":
    check_labels("../../resources/cup.mp4", "../../resources/cup.pickle")
