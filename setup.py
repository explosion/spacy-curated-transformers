from setuptools import setup, find_packages


def setup_package():
    setup(
        name="spacy-curated-transformers",
        packages=find_packages(),
    )


if __name__ == "__main__":
    setup_package()
