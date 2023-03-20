"""Includes utilities for downloading and preprocessing the raw dataset
obtained from https://www.mutopiaproject.org"""

import os
from rdflib import Graph


def download_ftp(unzip: bool = False):
    """Downloads all does basic processing on all .mid (and .rdf files) from
    the hello Mutopia Project FTP server. Additionally supports downloading and
    unzipping all files matching "*mids.zip". Depending on your connection,
    this should take around 2 hours to run.

    BEWARE: Running this function unzip=True will recursively unzip and remove
    all zip files (and empty directories). I would suggest running this
    function in a empty directory to be safe.

    Note this function is designed for use on Unix (Debian) machines. If you
    require the raw files otherwise, please send me an email."""

    assert os.listdir(".") == [], "Current directory is not empty."

    # Cd to new directory
    os.system("""mkdir mutopia""")

    # Gets all relevant files from ftp
    os.system(
        """wget -r -l 10 -nc -p -A rdf,mid,"*mids.zip" ftp://www.mutopiaproject.org/Mutopia/"""
    )

    # Move folder, remove old folder
    os.system("mv www.mutopiaproject.org/Mutopia/* mutopia")
    os.system("""rm -r www.mutopiaproject.org""")

    # Unzip and remove unneeded zip files and directories
    if unzip is True:
        os.system(
            """find mutopia -name "*.zip" | while read filename; do unzip -j -o -d "`dirname "$filename"`" "$filename"; done;"""
        )
        os.system("""find mutopia -type d -empty -delete""")
        os.system("""find mutopia -name "*.zip" -type f -delete""")


def parse_metadata(dir: str, meta_tags: list | None = None):
    """Parses .rdf metadata of all files located in dir.

    Note that this function is very hacky. I'm not an expert on rdf or xml. It
    also will only work for the .rdf files obtained using download_ftp().

    Args:
        dir (str): Directory to search for .rdf file.
        meta_tags (list[str], optional): list of meta data tags to search for
            and return in .rdf files. If not provided, the default meta_tags
            will be used. These are 'title', 'composer', 'opus', 'for',
            'style', 'licence'.

    Returns:
        dict: Relevant metadata located in .rdf file (xml).
    """

    def _parse_rdf(file: str, meta_tags: list[str]):
        """Internal function for extracting meta data from .rdf file.

        Args:
            file (str): File to parse.
            meta_tags: (list[str]): Metadata tags to extract.

        Returns:
            dict: Extracted metadata.
            int: Number of .rdf files present in dir. Used for error handling.
        """

        g = Graph()
        g.parse(file, format="xml")

        res = {}
        for s, p, o in g:
            tag = p.rsplit("/", 1)[-1]
            if tag in meta_tags:
                res[tag] = str(o)

        return res

    if meta_tags is None:
        meta_tags = [
            "title",
            "composer",
            "opus",
            "for",
            "style",
            "licence",
        ]

    rdf_count = 0
    for file in os.listdir(dir):
        if file.endswith(".rdf"):
            meta_data = _parse_rdf(file, meta_tags)
            rdf_count += 1

    return meta_data, rdf_count
