import json
from pathlib import Path


def main():
    path = Path(
        "data/processed/nvidia_2026_10k.json"
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    headings = [
        element
        for element in data["elements"]
        if element["element_type"] == "heading"
    ]

    print(f"Total headings: {len(headings)}\n")

    for heading in headings:
        print(
            f"Page {heading['page_number']}: "
            f"{heading['text']}"
        )


if __name__ == "__main__":
    main()